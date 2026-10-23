import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.graphx.{Edge, Graph, VertexId}
import org.apache.spark.graphx.lib.LabelPropagation

object MovieGraphAnalysis {
  private val MovieVertexOffset: Long = 1000000000000L

  private def parseRating(line: String): Option[(Long, Long, Double)] = {
    val fields = line.split(",", -1)
    if (fields.length < 3) None
    else {
      try {
        val userId = fields(0).trim.toLong
        val movieId = fields(1).trim.toLong
        val rating = fields(2).trim.toDouble
        if (userId <= 0 || movieId <= 0 || rating < 0.5 || rating > 5.0) None
        else Some((userId, MovieVertexOffset + movieId, rating))
      } catch {
        case _: NumberFormatException => None
      }
    }
  }

  def main(args: Array[String]): Unit = {
    require(args.length == 2, "Usage: MovieGraphAnalysis <ratings.csv> <output-directory>")
    val conf = new SparkConf().setAppName("movie-graphx-analysis")
    val sc = new SparkContext(conf)
    try {
      val rows = sc.textFile(args(0)).filter(line => !line.startsWith("userId,"))
        .flatMap(parseRating).cache()
      val vertices = rows.flatMap { case (userId, movieVertex, _) =>
        Seq((userId: VertexId, "user"), (movieVertex: VertexId, "movie"))
      }.distinct()
      val edges = rows.flatMap { case (userId, movieVertex, rating) =>
        Seq(Edge(userId: VertexId, movieVertex: VertexId, rating), Edge(movieVertex: VertexId, userId: VertexId, rating))
      }
      val graph = Graph(vertices, edges, "unknown").cache()

      // Label propagation provides a scalable structural community signal on the bipartite graph.
      val communities = LabelPropagation.run(graph, 5)
      val communityOutput = communities.vertices.join(graph.vertices).map { case (id, (label, kind)) =>
        s"$kind,$id,$label"
      }
      val popularMovies = graph.inDegrees.join(graph.vertices)
        .filter { case (_, (_, kind)) => kind == "movie" }
        .map { case (id, (degree, _)) => (id, degree) }
        .sortBy({ case (_, degree) => -degree })
        .map { case (id, degree) => s"${id - MovieVertexOffset},$degree" }
      val userDegrees = graph.inDegrees.join(graph.vertices)
        .filter { case (_, (_, kind)) => kind == "user" }
        .map { case (id, (degree, _)) => s"$id,$degree" }

      communityOutput.saveAsTextFile(s"${args(1)}/communities")
      popularMovies.saveAsTextFile(s"${args(1)}/popular_movies")
      userDegrees.saveAsTextFile(s"${args(1)}/user_degrees")
      println(s"vertices=${graph.vertices.count()} edges=${graph.edges.count()} partitions=${graph.vertices.partitions.length}")
    } finally {
      sc.stop()
    }
  }
}
