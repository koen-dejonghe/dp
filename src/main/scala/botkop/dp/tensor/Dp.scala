package botkop.dp.tensor

import botkop.{numsca => ns}
import botkop.numsca.Tensor

case class Variable(var data: Tensor, f: Option[Function] = None) {
  lazy val g: Tensor = ns.zerosLike(data)

  def backward(gradOutput: Tensor): Unit = {
    g += gradOutput
    for (gf <- f) gf.backward(gradOutput)
  }

  def backward(): Unit = backward(ns.ones(data.shape))

  import Function._
  def +(v: Variable): Variable = add(this, v)
  def *(v: Variable): Variable = mul(this, v)
}

trait Function {
  def forward(): Variable
  def backward(g: Tensor): Unit
}

object Function {
  case class Add(a: Variable, b: Variable) extends Function {
    override def forward(): Variable = Variable(a.data + b.data, Some(this))
    override def backward(g: Tensor): Unit = {
      a.backward(g)
      b.backward(g)
    }
  }
  def add(a: Variable, b: Variable): Variable = Add(a, b).forward()

  case class Mul(a: Variable, b: Variable) extends Function {
    override def forward(): Variable = Variable(a.data * b.data, Some(this))
    override def backward(g: Tensor): Unit = {
      a.backward(b.data * g)
      b.backward(a.data * g)
    }
  }
  def mul(a: Variable, b: Variable): Variable = Mul(a, b).forward()
}

abstract class Optimizer(parameters: Seq[Variable]) {
  def step(): Unit
  def zeroGrad(): Unit = parameters.foreach(p => p.g := 0)
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

abstract class Module(localParameters: Seq[Variable]) {

  // todo also obtain local parameters through introspection
  // and test this

  // by default, obtain submodules through introspection
  lazy val subModules: Seq[Module] =
    this.getClass.getDeclaredFields.flatMap { f =>
      f setAccessible true
      f.get(this) match {
        case module: Module => Some(module)
        case _              => None
      }
    }

  def parameters: Seq[Variable] =
    localParameters ++ subModules.flatMap(_.parameters)

  def gradients: Seq[Tensor] = parameters.map(_.g)
  def zeroGrad(): Unit = parameters.foreach(p => p.g := 0)
  def forward(x: Variable): Variable
  def apply(x: Variable): Variable = forward(x)
}

class Dp {}
