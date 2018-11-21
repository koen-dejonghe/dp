package botkop.dp.tensor

import scala.annotation.tailrec

case class Gate(m: Module, o: Optimizer, lf: (Variable, Variable) => Variable) {

  def learn(xys: Iterable[(Variable, Variable)],
            learnUntil: (Int, Double) => Boolean,
            logEvery: Int): (Int, Double) = {
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

  def learn(xys: Iterable[(Variable, Variable)],
            numEpochs: Int,
            logEvery: Int = 1): (Int, Double) =
    learn(xys, (i, _) => i >= numEpochs, logEvery)

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
    l.data.squeeze()
  }

  def apply(x: Variable): Variable = m(x)
}
