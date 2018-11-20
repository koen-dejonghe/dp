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

  def +(v: Variable): Variable = Add(this, v).forward()
  def *(v: Variable): Variable = Mul(this, v).forward()
}

trait Function {
  def forward(): Variable
  def backward(g: Double): Unit
}

case class Add(a: Variable, b: Variable) extends Function {
  override def forward(): Variable = Variable(a.data + b.data, Some(this))
  override def backward(g: Double): Unit = {
    a.backward(g)
    b.backward(g)
  }
}

case class Mul(a: Variable, b: Variable) extends Function {
  override def forward(): Variable = Variable(a.data * b.data, Some(this))
  override def backward(g: Double): Unit = {
    a.backward(b.data * g)
    b.backward(a.data * g)
  }
}

abstract class Optimizer(parameters: Seq[Variable]) {
  def step(): Unit
  def zeroGrad(): Unit = parameters.foreach(p => p.g = 0)
}

case class SGD(parameters: Seq[Variable], lr: Double)
    extends Optimizer(parameters) {
  override def step(): Unit = parameters.foreach { p =>
    p.data -= p.g * lr
  }
}

abstract class Module {
  def parameters: Seq[Variable]
  def gradients: Seq[Double] = parameters.map(_.g)
  def zeroGrad(): Unit = parameters.foreach(p => p.g = 0)
  def forward(x: Variable): Variable
  def apply(x: Variable): Variable = forward(x)
}

case class Net(m: Module, mlf: (Variable, Variable) => Function, o: Optimizer) {

  def learnUntil(xys: Iterator[(Variable, Variable)],
                 condition: (Int, Double) => Boolean): (Int, Double) = {
    @tailrec
    def lu(i: Int = 1): (Int, Double) = {
      val l = learn(xys)
      println(l)
      if (!condition(i, l))
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
    val l = mlf(yHat, y).forward()
    l.backward()
    o.step()
    l.data
  }

  def apply(x: Variable): Variable = m(x)
}

object DpApp extends App {

  def function(x: Double): Double = x / 2 + 3
  val numSamples = 10

  val xys = (1 to numSamples) map { _ =>
    val x = Random.nextDouble()
    val y = function(x)
    (Variable(x), Variable(y))
  }

  case class SimpleLoss(actual: Variable, target: Variable) extends Function {
    val diff: Double = actual.data - target.data
    override def forward(): Variable =
      Variable(diff, Some(this))
    override def backward(g: Double): Unit =
      actual.backward(diff)
  }

  def makeLossFunction(x: Variable, y: Variable): Function = SimpleLoss(x, y)

  val m: Module = new Module() {
    val w: Variable = Variable(Random.nextDouble())
    val b: Variable = Variable(0)
    override def parameters: Seq[Variable] = Seq(w, b)
    override def forward(x: Variable): Variable =
      x * w + b
  }

  val o = SGD(m.parameters, 1e-4)

  val n = Net(m, makeLossFunction, o)

  n.learnUntil(xys.iterator, (i, _) => i > 1000)

  val x = 11
  val yh = n(Variable(x)).data
  val y = function(x)
  println(s"x=$x f(x)=$y y^=$yh")

  /*
  val printEvery = 100
  for (it <- 1 to 10000) {
    val loss = xs.zip(ys).foldLeft(0.0) {
      case (z, (x, y)) =>
        m.zeroGrad()
        val yHat = m(x)
        val l = Loss(yHat, y).forward()
        l.backward()
        o.step()
        z + l.data
    }

    if (it % printEvery == 0) {
      val x = 11
      val yh = m(Variable(x)).data
      val y = function(x)
      val l = loss / (numSamples * printEvery)
      println(s"$it x=$x f(x)=$y y^=$yh ($l)")
    }
  }
 */

}
