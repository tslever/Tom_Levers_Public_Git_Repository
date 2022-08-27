//----------------------------------------------------------------------------
// UseGraph.java             by Dale/Joyce/Weems                    Chapter 10
//
// Examples of uses of the Graph ADT.
//----------------------------------------------------------------------------
package ch10.apps;

import ch04.queues.*;
import ch02.stacks.*;
import ch10.graphs.*;    
import ch09.priorityQueues.*;
import support.Flight;

public class UseGraph
{
private static void shortestPaths(WeightedGraphInterface<String> graph,
                                  String startVertex  )
// Writes the shortest distance from startVertex to every 
// other reachable vertex in graph.
{
  Flight flight;
  Flight saveFlight;         // for saving on priority queue
  int minDistance;
  int newDistance;

  PriQueueInterface<Flight> pq = new HeapPriQ<Flight>(20);   // Assume at most 20 vertices
  String vertex;
  QueueInterface<String> vertexQueue = new LinkedQueue<String>();
  
  graph.clearMarks();
  saveFlight = new Flight(startVertex, startVertex, 0);
  pq.enqueue(saveFlight);

  System.out.println("Last Vertex   Destination   Distance");
  System.out.println("------------------------------------");
 
  do
  {
    flight = pq.dequeue();
    if (!graph.isMarked(flight.getToVertex()))
    {
      graph.markVertex(flight.getToVertex());
      System.out.println(flight);
      flight.setFromVertex(flight.getToVertex());
      minDistance = flight.getDistance();
      vertexQueue = graph.getToVertices(flight.getFromVertex());
      while (!vertexQueue.isEmpty())
      {
        vertex = vertexQueue.dequeue();
        if (!graph.isMarked(vertex))
        {
          newDistance = minDistance 
                        + graph.weightIs(flight.getFromVertex(), vertex);
          saveFlight = new Flight(flight.getFromVertex(), vertex, newDistance);
          pq.enqueue(saveFlight);
        }
      }
    }
  } while (!pq.isEmpty());
  System.out.println();
  System.out.println("The unreachable vertices are:");
  vertex = graph.getUnmarked();
  while (vertex != null)
  {
    System.out.println(vertex);
    graph.markVertex(vertex);
    vertex = graph.getUnmarked();
  }
  System.out.println();
}

private static boolean isPathDF(WeightedGraphInterface<String> graph,
                                String startVertex, 
                                String endVertex    )
// Returns true if a path exists on graph, from startVertex to endVertex; 
// otherwise returns false. Uses depth-first search algorithm.
{
  StackInterface<String> stack = new LinkedStack<String>();
  QueueInterface<String> vertexQueue = new LinkedQueue<String>();
 
  boolean found = false;
  String currVertex;      // vertex being processed
  String adjVertex;       // adjacent to currVertex
 
  graph.clearMarks();
  graph.markVertex(startVertex);
  stack.push(startVertex);
  
  do
  {
    currVertex = stack.top();
    stack.pop();
    System.out.println(currVertex);
    if (currVertex.equals(endVertex))
       found = true;
    else
    {
      vertexQueue = graph.getToVertices(currVertex); 
      while (!vertexQueue.isEmpty())
      {
        adjVertex = vertexQueue.dequeue();
        if (!graph.isMarked(adjVertex))
        {
          graph.markVertex(adjVertex);
          stack.push(adjVertex);
        }
      }
    }
  } while (!stack.isEmpty() && !found);
  
  return found;
}

private static boolean isPathBF(WeightedGraphInterface<String> graph,
                                String startVertex, 
                                String endVertex    )

// Returns true if a path exists on graph, from startVertex to endVertex; 
// otherwise returns false. Uses breadth-first search algorithm.

{
  QueueInterface<String> queue = new LinkedQueue<String>();
  QueueInterface<String> vertexQueue = new LinkedQueue<String>();
 
  boolean found = false;
  String currVertex;      // vertex being processed
  String adjVertex;       // adjacent to currVertex

  graph.clearMarks();
  graph.markVertex(startVertex);
  queue.enqueue(startVertex);
  
  do
  {
    currVertex = queue.dequeue();
    System.out.println(currVertex);
    if (currVertex.equals(endVertex))
       found = true;
    else
    {
      vertexQueue = graph.getToVertices(currVertex); 
      while (!vertexQueue.isEmpty())
      {
        adjVertex = vertexQueue.dequeue();
        if (!graph.isMarked(adjVertex))
        {
          graph.markVertex(adjVertex);
          queue.enqueue(adjVertex);
        }
      }
    }
  } while (!queue.isEmpty() && !found);
  
  return found;
}


  public static void main(String[] args) 
  {
    // Create and analyze grph in Figure 10.3
    System.out.println("Creating graph in figure 10.3");
    WeightedGraphInterface<String> graph = new WeightedGraph<String>();
    String s0 = new String("Atlanta   ");
    String s1 = new String("Austin    ");
    String s2 = new String("Chicago   ");
    String s3 = new String("Dallas    ");
    String s4 = new String("Denver    ");
    String s5 = new String("Houston   ");
    String s6 = new String("Washington");

    graph.addVertex(s0);
    graph.addVertex(s1);
    graph.addVertex(s2);
    graph.addVertex(s3);
    graph.addVertex(s4);
    graph.addVertex(s5);
    graph.addVertex(s6);

    graph.addEdge(s0, s5, 800);
    graph.addEdge(s0, s6, 600);
    graph.addEdge(s1, s3, 200);
    graph.addEdge(s1, s5, 160);
    graph.addEdge(s2, s4, 1000);
    graph.addEdge(s3, s1, 200);
    graph.addEdge(s3, s2, 900);
    graph.addEdge(s3, s4, 780);
    graph.addEdge(s4, s0, 1400);
    graph.addEdge(s4, s2, 1000);
    graph.addEdge(s5, s0, 800);
    graph.addEdge(s6, s0, 600);
    graph.addEdge(s6, s3, 1300);

    boolean result;

    System.out.println("Determining path using depth first ...");
    System.out.println();
    
    System.out.println(s1 + " to " + s2);
    result = isPathDF(graph, s1, s2);
    System.out.println(result);
    System.out.println();
    
    System.out.println(s1 + " to " + s6);
    result = isPathDF(graph, s1, s6);
    System.out.println(result);
    System.out.println();
    
    System.out.println(s3 + " to " + s1);
    result = isPathDF(graph, s3, s1);
    System.out.println(result);
    System.out.println();
    
    System.out.println(s0 + " to " + s4);
    result = isPathDF(graph, s0, s4);
    System.out.println(result);
    System.out.println();

    
    System.out.println(s6 + " to " + s3);
    result = isPathDF(graph, s6, s3);
    System.out.println(result);
    System.out.println();
    
    System.out.println(s6 + " to " + s1);
    result = isPathDF(graph, s6, s1);
    System.out.println(result);
    System.out.println();
    
    System.out.println();
    System.out.println("Determining path using breadth first ...");
    System.out.println();
    
    System.out.println(s1 + " to " + s2);
    result = isPathBF(graph, s1, s2);
    System.out.println(result);
    System.out.println();
    
    System.out.println(s1 + " to " + s6);
    result = isPathBF(graph, s1, s6);
    System.out.println(result);
    System.out.println();
    
    System.out.println(s3 + " to " + s1);
    result = isPathBF(graph, s3, s1);
    System.out.println(result);
    System.out.println();
    
    System.out.println(s0 + " to " + s4);
    result = isPathBF(graph, s0, s4);
    System.out.println(result);
    System.out.println();
    
    System.out.println(s6 + " to " + s3);
    result = isPathBF(graph, s6, s3);
    System.out.println(result);
    System.out.println();
    
    System.out.println(s6 + " to " + s1);
    result = isPathBF(graph, s6, s1);
    System.out.println(result);
    
    System.out.println();
    System.out.println();
    System.out.println("Shortest paths starting at " + s6);
    shortestPaths(graph, s6);
    
    System.out.println();
    System.out.println();
    System.out.println("Shortest paths starting at " + s4);
    shortestPaths(graph, s4);

    System.out.println();
    System.out.println();
    
    
    // Create and analyze graph in Figure 10.x
    System.out.println("Creating graph in figure 10.x");
    graph = new WeightedGraph<String>();
    s0 = new String("Atlanta   ");
    s1 = new String("Austin    ");
    s2 = new String("Chicago   ");
    s3 = new String("Dallas    ");
    s4 = new String("Denver    ");
    s5 = new String("Houston   ");
    s6 = new String("Washington");

    graph.addVertex(s0);
    graph.addVertex(s1);
    graph.addVertex(s2);
    graph.addVertex(s3);
    graph.addVertex(s4);
    graph.addVertex(s5);
    graph.addVertex(s6);

    graph.addEdge(s0, s5, 800);
    graph.addEdge(s0, s6, 600);
    graph.addEdge(s1, s3, 200);
    graph.addEdge(s1, s5, 160);
    graph.addEdge(s2, s4, 1000);
    graph.addEdge(s3, s1, 200);
    graph.addEdge(s3, s2, 900);
    graph.addEdge(s3, s4, 780);
    graph.addEdge(s4, s0, 1400);
    graph.addEdge(s4, s2, 1000);
    graph.addEdge(s5, s0, 800);
    graph.addEdge(s6, s0, 600);
//    graph.addEdge(s6, s3, 1300);

    System.out.println("Determining path using depth first ...");
    System.out.println();
   
    System.out.println(s1 + " to " + s2);
    result = isPathDF(graph, s1, s2);
    System.out.println(result);
    System.out.println();
    
    System.out.println(s1 + " to " + s6);
    result = isPathDF(graph, s1, s6);
    System.out.println(result);
    System.out.println();
    
    System.out.println(s6 + " to " + s5);
    result = isPathDF(graph, s6, s5);
    System.out.println(result);
    System.out.println();
    
    System.out.println(s6 + " to " + s3);
    result = isPathDF(graph, s6, s3);
    System.out.println(result);
    System.out.println();
    
    System.out.println(s6 + " to " + s1);
    result = isPathDF(graph, s6, s1);
    System.out.println(result);
    System.out.println();
    
    System.out.println();
    System.out.println("Determining path using breadth first ...");
    System.out.println();
    
    System.out.println(s1 + " to " + s2);
    result = isPathBF(graph, s1, s2);
    System.out.println(result);
    System.out.println();
    
    System.out.println(s1 + " to " + s6);
    result = isPathBF(graph, s1, s6);
    System.out.println(result);
    System.out.println();
    
    System.out.println(s6 + " to " + s5);
    result = isPathBF(graph, s6, s5);
    System.out.println(result);
    System.out.println();
    
    System.out.println(s6 + " to " + s3);
    result = isPathBF(graph, s6, s3);
    System.out.println(result);
    System.out.println();
    
    System.out.println(s6 + " to " + s1);
    result = isPathBF(graph, s6, s1);
    System.out.println(result);
    
    System.out.println();
    System.out.println();
    System.out.println("Shortest paths starting at " + s6);
    shortestPaths(graph, s6);
    
    System.out.println();
    System.out.println();
    System.out.println("Shortest paths starting at " + s4);
    shortestPaths(graph, s4);

    System.out.println();


  } 
} 