package botkop.dp.lantern

import botkop.{numsca => ns}
import botkop.numsca.Tensor
import XTensor._

import scala.language.implicitConversions
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
  def sofmaxLoss(actual: Variable, target: Variable) = shift { (k: Variable => Unit) =>
    val x: Tensor = actual.x
    val y: Tensor = target.x.T

    val shiftedLogits: Tensor = x - ns.max(x, axis = 1)
    val z: Tensor = ns.sum(ns.exp(shiftedLogits), axis = 1)
    val logProbs: Tensor = shiftedLogits - ns.log(z)
    val n: Int = x.shape.head
    val loss: Double = -ns.sum(logProbs(ns.arange(n), y)) / n

    k(loss)

    val dx = ns.exp(logProbs)
    dx(ns.arange(n), y) -= 1
    dx /= n
    actual.d +:= dx
  }

}
