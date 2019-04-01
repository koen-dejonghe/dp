package botkop.dp.lantern

import botkop.{numsca => ns}
import botkop.numsca.Tensor

import scala.language.implicitConversions
import scala.util.continuations.{cps, reset, shift}

object NumscaLantern extends App {

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

  implicit def t2xt(t: Tensor): XTensor = XTensor(t)

  case class Variable(x: Tensor, d: Tensor) {

    def +(that: Variable) = shift { (k: Variable => Unit) =>
      val y = Variable(x + that.x)
      k(y)
      this.d +:= y.d
      that.d +:= y.d
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

  }

  object Variable {
    def apply(t: Tensor): Variable = Variable(t, ns.zerosLike(t))
    def apply(d: Double): Variable = Variable(Tensor(d))
    implicit def doubleToVariable(d: Double): Variable = Variable(d)
    implicit def tensorToVariable(t: Tensor): Variable = Variable(t)
  }

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

  val x = Variable(ns.arange(12).reshape(3, 4))
  val y = Variable(ns.arange(8).reshape(4, 2))
  // val z = x.dot(y)

  val g = ns.arange(6).reshape(3, 2)

  def grad2(f: Variable => Variable @cps[Unit])(x: Variable): Variable = {
    reset { f(x).d := g }
    x.d
  }
  val df2: Variable => Variable = grad2(x => x dot y)

  val dz = df2(x)
  println("========================")
  println(dz)
  println(x.d)
  println(y.d)


  x.d := 0.0
  y.d := 0.0
  def g2(f: (Variable, Variable) => Variable @cps[Variable])(x: Variable, y: Variable, g: Tensor): Variable = {
    reset {
      val r = f(x, y)
      r.d := g
      r
    }
  }

  val df3 = g2((x0, y0) => x0 dot y0)
  df3(x, y, g)
  println("========================")
  println(x.d)
  println(y.d)

}
