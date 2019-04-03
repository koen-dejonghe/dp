package botkop.dp.lantern

import botkop.{numsca => ns}
import botkop.numsca.Tensor
import XTensor._

import scala.language.implicitConversions
import scala.util.continuations.{cps, reset, shift}

object NumscaLantern extends App {

  /* function we're trying to guess */
  def unknownFunction(x: Tensor): Tensor = ns.sum(x * x, axis = 1)

  val numBatches = 1000
  val numSamples = 3
  val numFeatures = 4
  val lr = 1e-2

  /* training set */
  val trainingSet = (1 to numBatches) map { _ =>
    val x = ns.randn(numSamples, numFeatures)
    val y = unknownFunction(x)
    (x, y)
  }

  /* test set */
  val testX = ns.randn(10, numFeatures)
  val testY = unknownFunction(testX)

  /* parameters of the network */
  val w0: Variable = Variable(ns.randn(100, numFeatures) * math.sqrt(2.0 / 100))
  val b0: Variable = Variable(ns.zeros(1, 100))

  val w1: Variable = Variable(ns.randn(4, 100) * math.sqrt(2.0 / 4))
  val b1: Variable = Variable(ns.zeros(1, 4))

  /* neural net definition */
  def net(x: Variable): Variable @cps[Unit] =
    (x.dot(w0.t) + b0).relu.dot(w1.t) + b1

  /* loss function */
  type LossFunction = (Tensor, Tensor) => Tensor
  def loss(expected: Tensor, actual: Tensor): Tensor = actual - expected

  /* training function */
  def train(f: Variable => Variable @cps[Unit],
            loss: LossFunction)(x: Tensor, y: Tensor): Unit =
    reset {
      val yHat = f(x)
      yHat.d := loss(y, yHat.x)
    }

  def eval(f: Variable => Variable @cps[Unit])(x: Tensor, y: Tensor): Unit =
    reset {
      val yHat = f(x)
      val score = ns.sum((testY - yHat.x) ** 2) / testX.shape(0)
      println(score)
    }

  /* training loop */
  for (epoch <- 1 to 10) {
    trainingSet foreach {
      case (x0, y0) =>
        train(net, loss)(x0, y0)

        w0.x -= w0.d * lr
        b0.x -= b0.d * lr
        w0.d := 0
        b0.d := 0

        w1.x -= w1.d * lr
        b1.x -= b1.d * lr
        w1.d := 0
        b1.d := 0
    }

    eval(net)(testX, testY)
  }

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
    this.d += y.d dot that.x.T
    that.d += this.x.T dot y.d
  }

  def t = shift { (k: Variable => Unit) =>
    val r = Variable(x.transpose())
    k(r)
    this.d += r.d.transpose()
  }

  def relu = shift { (k: Variable => Unit) =>
    val r = Variable(ns.maximum(x, 0.0))
    k(r)
    this.d += r.d * (x > 0.0)
  }
}

object Variable {
  def apply(d: Double): Variable = Variable(Tensor(d))
  implicit def doubleToVariable(d: Double): Variable = Variable(d)
  implicit def tensorToVariable(t: Tensor): Variable = Variable(t)
}

/* tensor extension to allow for 'unbroadcasting', ie. shaping another tensor like this one. */
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

