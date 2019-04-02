package botkop.dp.lantern

import botkop.{numsca => ns}
import botkop.numsca.Tensor
import XTensor._

import scala.language.implicitConversions
import scala.util.Random
import scala.util.continuations.{cps, reset, shift}

object NumscaLanternGrad extends App {

  def grad(f: Variable => Variable @cps[Unit])(x: Tensor): Tensor = {
    val z = Variable(x)
    reset { f(z).d := 1.0 }
    z.d
  }

  val df: Tensor => Tensor = grad((x: Variable) => x * 2.0 + x * x * x)

  assert(List.fill(3)(ns.randn(3, 3)).forall { x =>
    val dx = df(x)
    val expected = 2 + 3 * x * x

    println("dx = " + dx)
    println("x = " + x)
    println("expected = " + expected)

    dx.sameShape(expected) && math.abs(ns.sum(expected) - ns.sum(dx)) < 1e-3

  // dx == x2

  })

  // val z = x.dot(y)

  val x = Variable(ns.arange(12).reshape(3, 4))
  val y = Variable(ns.arange(8).reshape(4, 2))
  val g = ns.arange(6).reshape(3, 2)

  def grad2(f: Variable => Variable @cps[Unit])(x: Variable): Variable = {
    reset {
      val r = f(x)
      r.d := g
    }
    x
  }
  val df2: Variable => Variable = grad2 { _ =>
    val z = x dot y
    z
  }

  val dz = df2(x)
  println("========================")
  println(dz)
  println(x.d)
  println(y.d)

  x.d := 0.0
  y.d := 0.0

  {
    reset {
      val z = x dot y
      val gt = ns.arange(6).reshape(3, 2)
      z.d := gt
    }
  }

  println("========================")
  println(x.d)
  println(y.d)
  /*
  x.d := 0.0
  y.d := 0.0
  def g2(f: (Variable, Variable) => Variable @cps[Unit])(x: Variable, y: Variable, g: Tensor): Variable = {
    reset {
      val r = f(x, y)
      r.d := g
      r
    }
  }

  val df3: (Variable, Variable, Tensor) => Variable  = g2((x0, y0) => x0 dot y0)
  df3(x, y, g)
  println("========================")
  println(x.d)
  println(y.d)
 */
}

object NumscaLantern extends App {

  def function(x: Tensor): Tensor = ns.sum(x * 5, axis = 1)

  val numBatches = 10
  val numSamples = 3
  val numFeatures = 4
  val lr = 1e-2

  val xys = (1 to numBatches) map { _ =>
    val x = ns.randn(numSamples, numFeatures)
    val y = function(x)
    (x, y)
  }

  val w0: Variable = Variable(ns.randn(10, numFeatures))
  val b0: Variable = Variable(ns.zeros(1, 10))

  val w1: Variable = Variable(ns.randn(1, 10))
  val b1: Variable = Variable(ns.zeros(1, 1))

  // def net(x: Tensor) = (Variable(x) dot w0.t + b0) dot w1.t + b1
  def net(x: Tensor) = {
    (Variable(x) dot w0.t) + b0
  }

  for (epoch <- 1 to 100) {
    xys foreach {
      case (x0, y0) =>
        reset {
//          val r = (Variable(x0).dot(w0.t)) + b0
//          println(r.shape)
          val r = net(x0)
          val g = y0 - r.x
          r.d -= g
        }

        w0.x -= w0.d * lr
        b0.x -= b0.d * lr
        w0.d := 0
        b0.d := 0
//
//        w1.x -= w1.d * lr
//        b1.x -= b1.d * lr
//        w1.d := 0
//        b1.d := 0
    }

    reset {
      val x = ns.randn(numSamples, numFeatures)
      val y = function(x)
      val yHat = net(x)
      println(ns.sum((y - yHat.x) ** 2))
    }
  }
//  val x = ns.randn(numSamples, numFeatures)
//  val y = function(x)
//  val yHat = x dot w.x.transpose() + b.x
//
//  println(x)
//  println(y)
//  println(yHat)

}

case class XTensor(t: Tensor) {
  def unbroadcast(target: Tensor): Tensor =
    if (target.shape sameElements t.shape)
      target
    else
      t.shape.zip(target.shape).zipWithIndex.foldLeft(target) {
        case (z: Tensor, ((oi, ni), i)) =>
          if (oi == ni)
            z
          else if (oi == 1)
            ns.sum(z, axis = i)
          else
            throw new Exception(
              s"unable to unbroadcast shape ${target.shape.toList} to ${t.shape.toList}")
      }

  def +:=(that: Tensor): Unit = t += unbroadcast(that)
}
object XTensor {
  implicit def t2xt(t: Tensor): XTensor = XTensor(t)
}

case class Variable(x: Tensor) {

  lazy val d: Tensor = ns.zerosLike(x)

  def shape: List[Int] = x.shape.toList

  def +(that: Variable) = shift { (k: Variable => Unit) =>
    val y = Variable(x + that.x)
    k(y)
    this.d +:= y.d
    that.d +:= y.d
  }

  def -(that: Variable) = shift { (k: Variable => Unit) =>
    val y = Variable(x - that.x)
    k(y)
    this.d +:= y.d
    that.d +:= -y.d
  }

  def *(that: Variable) = shift { (k: Variable => Unit) =>
    val y = Variable(x * that.x)
    k(y)
    this.d +:= that.x * y.d
    that.d +:= this.x * y.d
  }

  def dot(that: Variable) = shift { (k: Variable => Unit) =>
    val y = Variable(x dot that.x)
    k(y)
    this.d +:= y.d dot that.x.T
    that.d +:= this.x.T dot y.d
  }

  def t = shift { (k: Variable => Unit) =>
    val r = Variable(x.transpose())
    k(r)
    this.d += r.d.transpose()
  }
}

object Variable {
//  def apply(t: Tensor): Variable = Variable(t, ns.zerosLike(t))
  def apply(d: Double): Variable = Variable(Tensor(d))
  implicit def doubleToVariable(d: Double): Variable = Variable(d)
  implicit def tensorToVariable(t: Tensor): Variable = Variable(t)
}
