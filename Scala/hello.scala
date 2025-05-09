object HelloWorld {

  // Function to add two integers
  def add(a: Int, b: Int): Int = {
    a + b
  }

  def main(args: Array[String]): Unit = {
    // Output message
    println("Hello, Scala!")

    // Immutable variable
    val name = "Alice"
    println("Name: " + name)

    // Mutable variable
    var age = 25
    println("Age: " + age)

    // Function usage
    val sum = add(10, 20)
    println("Sum of 10 and 20 is: " + sum)

    // For loop
    println("Counting from 1 to 5:")
    for (i <- 1 to 5) {
      println(i)
    }
  }
}
