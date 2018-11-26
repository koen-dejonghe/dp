package botkop.dp.numsca

import botkop.numsca.Tensor
import botkop.{numsca => ns}

trait Function {
  def forward(): Variable
  def backward(g: Tensor): Unit
}

object Function {
  def add(a: Variable, b: Variable): Variable = Add(a, b).forward()
  def mul(a: Variable, b: Variable): Variable = Mul(a, b).forward()
  def dot(a: Variable, b: Variable): Variable = Dot(a, b).forward()
  def transpose(v: Variable): Variable = Transpose(v).forward()
  def relu(v: Variable): Variable = Relu(v).forward()
  def softmaxLoss(x: Variable, y: Variable): Variable =
    SoftmaxLoss(x, y).forward()
}

case class Add(a: Variable, b: Variable) extends Function {
  override def forward(): Variable = {
    Variable(a.data + b.data, Some(this))
  }
  override def backward(g: Tensor): Unit = {
    a.backward(g)
    b.backward(g)
  }
}

case class Mul(a: Variable, b: Variable) extends Function {
  override def forward(): Variable = Variable(a.data * b.data, Some(this))
  override def backward(g: Tensor): Unit = {
    a.backward(b.data * g)
    b.backward(a.data * g)
  }
}

case class Dot(v1: Variable, v2: Variable) extends Function {
  val w: Tensor = v1.data
  val x: Tensor = v2.data

  override def forward(): Variable = Variable(w dot x, Some(this))
  override def backward(g: Tensor): Unit = {
    val dw = g dot x.T
    val dx = w.T dot g
    v1.backward(dw)
    v2.backward(dx)
  }
}

case class Transpose(v: Variable) extends Function {
  override def forward(): Variable = Variable(v.data.transpose, Some(this))
  override def backward(g: Tensor): Unit =
    v.backward(g.transpose)
}

case class Relu(x: Variable) extends Function {
  override def forward(): Variable =
    Variable(ns.maximum(x.data, 0), Some(this))
  override def backward(g: Tensor): Unit = {
    x.backward(g * (x.data > 0))
  }
}

case class SoftmaxLoss(actual: Variable, target: Variable) extends Function {
  val x: Tensor = actual.data
  val y: Tensor = target.data.T

  val shiftedLogits: Tensor = x - ns.max(x, axis = 1)
  val z: Tensor = ns.sum(ns.exp(shiftedLogits), axis = 1)
  val logProbs: Tensor = shiftedLogits - ns.log(z)
  val n: Int = x.shape.head
  val loss: Double = -ns.sum(logProbs(ns.arange(n), y)) / n

  override def forward(): Variable = Variable(Tensor(loss), Some(this))

  override def backward(g: Tensor): Unit = {
    val dx = ns.exp(logProbs)
    dx(ns.arange(n), y) -= 1
    dx /= n
    actual.backward(dx)
  }
}
