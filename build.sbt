name := "dp"

version := "0.1"

scalaVersion := "2.12.7"

libraryDependencies += "be.botkop" %% "numsca" % "0.1.3"

// cps
autoCompilerPlugins := true
addCompilerPlugin("org.scala-lang.plugins" % "scala-continuations-plugin_2.12.0" % "1.0.3")
libraryDependencies += "org.scala-lang.plugins" % "scala-continuations-library_2.12" % "1.0.3"
scalacOptions += "-P:continuations:enable"

