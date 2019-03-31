package botkop.dp.numsca

import botkop.{numsca => ns}

object Dp extends App {

  import Module._
  import Function._
  import Optimizer._

  val numSamples = 16
  val numClasses = 5
  val nf1 = 40
  val nf2 = 20

  val input = Variable(ns.randn(numSamples, nf1))
  val target = Variable(ns.randint(numClasses, Array(numSamples, 1)))
  val xys = Seq((input, target))

  val m: Module = new Module {
    val fc1: Linear = linear(nf1, nf2)
    val fc2: Linear = linear(nf2, numClasses)
    override def forward(x: Variable): Variable = x ~> fc1 ~> relu ~> fc2
  }

  val o = sgd(m, 1e-1)
  val g = Gate(m, o, softmaxLoss)
  g.learn(xys, 100, 10)

}

object Dp1 extends App {

  def f(v: Variable): Variable = 2 * v + v * v * v

  val t = Variable(ns.randn(2, 3))
  val r = f(t)
  r.backward()
  println(t.g)

}
