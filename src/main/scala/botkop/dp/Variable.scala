package botkop.dp

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
  def -(v: Variable): Variable = Min(this, v).forward()
  def *(v: Variable): Variable = Mul(this, v).forward()
  def *(d: Double): Variable = MulConstant(this, d).forward()
  def **(d: Double): Variable = PowConstant(this, d).forward()

  def ~>(cf: Variable => Variable): Variable = cf(this)
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

case class Min(a: Variable, b: Variable) extends Function {
  override def forward(): Variable = Variable(a.data - b.data, Some(this))
  override def backward(g: Double): Unit = {
    a.backward(g)
    b.backward(-g)
  }
}

case class Mul(a: Variable, b: Variable) extends Function {
  override def forward(): Variable = Variable(a.data * b.data, Some(this))
  override def backward(g: Double): Unit = {
    a.backward(b.data * g)
    b.backward(a.data * g)
  }
}

case class MulConstant(v: Variable, d: Double) extends Function {
  override def forward(): Variable = Variable(v.data * d, Some(this))
  override def backward(gradOutput: Double): Unit = {
    val dv = gradOutput * d
    v.backward(dv)
  }
}

case class PowConstant(v: Variable, d: Double) extends Function {
  override def forward(): Variable = Variable(math.pow(v.data, d), Some(this))
  val cache: Double = d * math.pow(v.data, d - 1)
  override def backward(gradOutput: Double): Unit = {
    val dv = cache * gradOutput
    v.backward(dv)
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

object DpApp extends App {

  def function(x: Double): Double = 2 * x + 37
  val numSamples = 10

  val (xs, ys) = (1 to numSamples) map { i =>
    val x = i.toDouble
    val y = function(x)
    (Variable(x), Variable(y))
  } unzip

  case class Loss(actual: Variable, target: Variable) extends Function {
    val diff: Double = actual.data - target.data
    override def forward(): Variable =
      Variable(diff, Some(this))
    override def backward(g: Double /* ignored */ ): Unit =
      actual.backward(diff)
  }

  val m: Module = new Module() {
    val ws: Seq[Variable] = Seq.fill(2) { Variable(Random.nextDouble()) }
    val bs: Seq[Variable] = Seq.fill(2) { Variable(0) }
    override def parameters: Seq[Variable] = ws ++ bs
    override def forward(x: Variable): Variable = ws.zip(bs).foldLeft(x) {
      case (z, (w, b)) =>
        z + w + b
    }
  }

  val o = SGD(m.parameters, 1e-4)

  val printEvery = 100
  for (it <- 1 to 2000) {
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


}
