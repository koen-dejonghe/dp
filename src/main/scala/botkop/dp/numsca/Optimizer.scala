package botkop.dp.numsca

object Optimizer {
  def sgd(m: Module, lr: Double): SGD = SGD(m.parameters, lr)
}

abstract class Optimizer(parameters: Seq[Variable]) {
  def step(): Unit
  def zeroGrad(): Unit = parameters.foreach(p => p.g := 0)
}

case class SGD(parameters: Seq[Variable], lr: Double)
  extends Optimizer(parameters) {
  override def step(): Unit = parameters.foreach { p =>
    p.data -= p.g * lr
  }
}
