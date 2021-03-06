package botkop.dp.lantern

import scala.language.implicitConversions
import scala.util.Random

object Lantern extends App {

  implicit def doubleToNumF(d: Double): NumF = new NumF(d, 0)
  def grad(f: NumF => NumF)(x: Double): Double = {
    val y = f(new NumF(x, 1.0))
    y.d
  }

  val df: Double => Double = grad(x => 2 * x + x * x * x)

  assert(List.fill(10)(Random.nextDouble()).forall { x =>
    df(x) == 2 + 3 * x * x
  })

  class NumF(val x: Double, val d: Double) {
    def +(that: NumF) =
      new NumF(this.x + that.x, this.d + that.d)
    def *(that: NumF) =
      new NumF(this.x * that.x, this.d * that.x + that.d * this.x)
  }

}

import scala.util.continuations._

object LanternCPS extends App {

  class NumR(val x: Double, var d: Double) {
    def +(that: NumR) = shift { (k: NumR => Unit) =>
      val y = new NumR(x + that.x, 0.0)
      k(y)
      this.d += y.d
      that.d += y.d
    }

    def *(that: NumR)= shift { (k: NumR => Unit) =>
      val y = new NumR(x * that.x, 0.0)
      k(y)
      this.d += that.x * y.d
      that.d += this.x * y.d
    }
  }

  def grad(f: NumR => NumR @cps[Unit])(x: Double): Double = {
    val z = new NumR(x, 0.0)
    reset { f(z).d = 1.0 }
    z.d
  }

  implicit def doubleToNumR(d: Double): NumR = new NumR(d, 0)

  val df: Double => Double = grad(x => 2 * x + x * x * x)

  assert(List.fill(10)(Random.nextDouble()).forall { x =>
    df(x) == 2 + 3 * x * x
  })

}
