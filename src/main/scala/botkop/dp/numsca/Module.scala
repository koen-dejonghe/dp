package botkop.dp.numsca

import botkop.numsca.Tensor
import botkop.{numsca => ns}

abstract class Module {
  lazy val (subModules, localParameters) =
    this.getClass.getDeclaredFields
      .foldLeft(List.empty[Module], List.empty[Variable]) {
        case ((zm, zv), f) =>
          f setAccessible true
          f get this match {
            case m: Module   => (m :: zm, zv)
            case v: Variable => (zm, v :: zv)
            case _           => (zm, zv)
          }
      }

  lazy val parameters: Seq[Variable] =
    localParameters ++ subModules.flatMap(_.parameters)

  lazy val gradients: Seq[Tensor] = parameters.map(_.g)

  def zeroGrad(): Unit = parameters.foreach(p => p.g := 0)
  def forward(x: Variable): Variable
  def apply(x: Variable): Variable = forward(x)
}

object Module {
  def linear(inFeatures: Int, outFeatures: Int): Linear = {
    val w: Tensor =
      ns.randn(outFeatures, inFeatures) * math.sqrt(2.0 / outFeatures)
    val weights = Variable(w)
    val b: Tensor = ns.zeros(1, outFeatures)
    val bias = Variable(b)
    Linear(weights, bias)
  }
}

case class Linear(weights: Variable, bias: Variable) extends Module {
  override def forward(x: Variable): Variable =
    x.dot(weights.t()) + bias
}
