package botkop.dp.bigdl

import com.intel.analytics.bigdl.nn.{Linear, SoftmaxWithCriterion}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.util.Random

object DpApp extends App {

  /*
  val v = Tensor[Float](2,3)
  val u = Tensor[Float](3,5).fill(3)


  val a = Tensor[Float](T(
    T(1f, 2f, 3f),
    T(4f, 5f, 6f)))

  val b = Tensor(Array.range(0, 9).map(_.toDouble), Array(3, 3))
  println(b)

  val c = b.view(Array(1, 9))

  c.setValue(1, 2, 2345.0)
  println(c)
  println(b)

  val n = new Linear[Float](784, 100)

  println(n)

  val d = Tensor[Float](1024, 784).randn()
  val f: Tensor[Float] = n.updateOutput(d)

  */


  val lr = 1e-4f
  val numSamples = 160
  val numClasses = 10
  val nf1 = 400
  val nf2 = 200

  val input = Tensor[Float](numSamples, nf1).randn()
  val target = Tensor[Float](Array.fill(numSamples)(Random.nextInt(numClasses).toFloat + 1), Array(numSamples, 1))

  val fc1 = new Linear[Float](nf1, nf2)
  val fc2 = new Linear[Float](nf2, numClasses)
  val sm = new SoftmaxWithCriterion[Float]()

  for (_ <- 1 to 1000) {
    val r0 = fc1.updateOutput(input)
    val r1 = fc2.updateOutput(r0)
    val loss = sm.updateOutput(r1, target)
    println(loss)

    val dl = sm.updateGradInput(r1, target)
    val dr1 = fc2.updateGradInput(r0, dl)
    val dr0 = fc1.updateGradInput(input, dr1)

    fc2.accGradParameters(r0, dl)
    fc1.accGradParameters(input, dr1)

    fc1.gradWeight.mul(lr)
    fc2.gradWeight.mul(lr)

    fc1.weight.add(fc1.gradWeight)
    fc2.weight.add(fc2.gradWeight)

    fc1.gradWeight.zero()
    fc2.gradWeight.zero()
  }


}
