package botkop.dp.numsca

import botkop.numsca.Tensor
import botkop.{numsca => ns}

import scala.language.implicitConversions

case class Variable(data: Tensor,
                    f: Option[Function] = None,
                    name: Option[String] = None) {
  lazy val g: Tensor = ns.zerosLike(data)

  def shape: List[Int] = data.shape.toList

  def backward(): Unit = backward(ns.ones(data.shape))

  def backward(gradOutput: Tensor): Unit = {
    val ubg = unbroadcast(gradOutput)
    g += ubg
    for (gf <- f) gf.backward(ubg)
  }

  // todo maybe implement this only for functions that can cause broadcasting
  def unbroadcast(t: Tensor): Tensor = {
    val gs = t.shape
    val ds = data.shape
    if (gs.sameElements(ds))
      t
    else
      ds.zip(gs).zipWithIndex.foldLeft(t) {
        case (z: Tensor, ((oi, ni), i)) =>
          if (oi == ni)
            z
          else if (oi == 1)
            ns.sum(z, axis = i)
          else
            throw new Exception(
              s"unable to unbroadcast shape ${gs.toList} to $shape")
      }
  }

  def ~>(f: Variable => Variable): Variable = f(this)

  def +(v: Variable): Variable = Function.add(this, v)
  def *(v: Variable): Variable = Function.mul(this, v)
  def dot(v: Variable): Variable = Function.dot(this, v)
  def t(): Variable = Function.transpose(this)
}

object Variable {

  implicit def moduleApply[T <: Module](m: T): Variable => Variable =
    m.forward

  def apply(d: Double): Variable = Variable(Tensor(d))
  def apply(d: Double, name: Option[String]): Variable =
    Variable(Tensor(d), name = name)
}
