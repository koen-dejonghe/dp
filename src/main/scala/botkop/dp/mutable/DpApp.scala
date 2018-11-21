package botkop.dp.mutable

import scala.annotation.tailrec
import scala.language.postfixOps
import scala.util.Random

case class Variable(var data: Double, f: Option[Function] = None) {
  var g: Double = 0

  def backward(gradOutput: Double): Unit = {
    g += gradOutput
    for (gf <- f) gf.backward(gradOutput)
  }

  def backward(): Unit = backward(1)

  import Function._
  def +(v: Variable): Variable = add(this, v)
  def *(v: Variable): Variable = mul(this, v)
}

trait Function {
  def forward(): Variable
  def backward(g: Double): Unit
}

object Function {

  case class SimpleLoss(x: Variable, y: Variable) extends Function {
    val diff: Double = x.data - y.data
    override def forward(): Variable = Variable(diff, Some(this))
    override def backward(g: Double): Unit = x.backward(diff)
  }
  def simpleLoss(x: Variable, y: Variable): Variable =
    SimpleLoss(x, y).forward()

  case class Add(a: Variable, b: Variable) extends Function {
    override def forward(): Variable = Variable(a.data + b.data, Some(this))
    override def backward(g: Double): Unit = {
      a.backward(g)
      b.backward(g)
    }
  }
  def add(a: Variable, b: Variable): Variable = Add(a, b).forward()

  case class Mul(a: Variable, b: Variable) extends Function {
    override def forward(): Variable = Variable(a.data * b.data, Some(this))
    override def backward(g: Double): Unit = {
      a.backward(b.data * g)
      b.backward(a.data * g)
    }
  }
  def mul(a: Variable, b: Variable): Variable = Mul(a, b).forward()

}

abstract class Optimizer(parameters: Seq[Variable]) {
  def step(): Unit
  def zeroGrad(): Unit = parameters.foreach(p => p.g = 0)
}

object Optimizer {

  case class SGD(parameters: Seq[Variable], lr: Double)
      extends Optimizer(parameters) {
    override def step(): Unit = parameters.foreach { p =>
      p.data -= p.g * lr
    }
  }

  def sgd(m: Module, lr: Double): SGD = SGD(m.parameters, lr)
}

abstract class Module {
  def parameters: Seq[Variable]
  def gradients: Seq[Double] = parameters.map(_.g)
  def zeroGrad(): Unit = parameters.foreach(p => p.g = 0)
  def forward(x: Variable): Variable
  def apply(x: Variable): Variable = forward(x)
}

case class Net(m: Module, o: Optimizer, lf: (Variable, Variable) => Variable) {
  def learn(xys: Iterable[(Variable, Variable)],
            learnUntil: (Int, Double) => Boolean,
            logEvery: Int = 100): (Int, Double) = {
    @tailrec
    def lu(i: Int = 1): (Int, Double) = {

      val l = learn(xys.iterator)
      if (i % logEvery == 0) {
        println(s"$i $l")
      }

      if (!learnUntil(i, l))
        lu(i + 1)
      else
        (i, l)
    }
    lu()
  }

  def learn(xys: Iterator[(Variable, Variable)]): Double = {
    val (l, n) = xys.foldLeft(0.0, 0) {
      case ((zl, zi), (x, y)) =>
        (zl + learn(x, y), zi + 1)
    }
    l / n
  }

  def learn(x: Variable, y: Variable): Double = {
    o.zeroGrad()
    val yHat = m(x)
    val l = lf(yHat, y)
    l.backward()
    o.step()
    l.data
  }

  def apply(x: Variable): Variable = m(x)
}

object DpApp extends App {

  def function(x: Double): Double = x / 2 + 3
  val numSamples = 10
  val lr = 0.1

  val xys = (1 to numSamples) map { _ =>
    val x = Random.nextDouble()
    val y = function(x)
    (Variable(x), Variable(y))
  }

  val m: Module = new Module() {
    val w: Variable = Variable(Random.nextDouble())
    val b: Variable = Variable(0)
    override def parameters: Seq[Variable] = Seq(w, b)
    override def forward(x: Variable): Variable =
      x * w + b
  }

  val o = Optimizer.sgd(m, lr)

  val n = Net(m, o, Function.simpleLoss)

  def learnUntil(epoch: Int, loss: Double): Boolean =
    epoch > 10000

  n.learn(xys, learnUntil _)

  val x = 11
  val yh = n(Variable(x)).data
  val y = function(x)
  println(s"x=$x f(x)=$y y^=$yh")

}
