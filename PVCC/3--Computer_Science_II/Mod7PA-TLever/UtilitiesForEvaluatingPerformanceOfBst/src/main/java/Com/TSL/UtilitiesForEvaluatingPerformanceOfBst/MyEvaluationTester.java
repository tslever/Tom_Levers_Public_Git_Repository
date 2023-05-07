package Com.TSL.UtilitiesForEvaluatingPerformanceOfBst;


import java.util.Arrays;


public class MyEvaluationTester {
	
	public static void main(String[] var0) {
		
		MyIntBSTTree theBinarySearchTreeOfIntegers = new MyIntBSTTree();

		int[] theArrayOfIntegers = {80, 17, 81, 85, 55, 12, 45};
		System.out.println("The int array is: " + Arrays.toString(theArrayOfIntegers));
		
		System.out.println("Build a tree by calling buildTree(arr):");
		theBinarySearchTreeOfIntegers = theBinarySearchTreeOfIntegers.buildTree(theArrayOfIntegers);
		theBinarySearchTreeOfIntegers.display();
		
		System.out.println("Height: " + theBinarySearchTreeOfIntegers.height());
					
		int theIntegerToFindInTheTree = 200;
		System.out.println(
			"Number of comparisons to search " + theIntegerToFindInTheTree + ": " +
			theBinarySearchTreeOfIntegers.comparisons(theIntegerToFindInTheTree)
		);
		
		System.out.println("\n-----------------------------------\n");
		
		System.out.println("Build a balanced tree by calling buildBalancedTree(arr):");
		MyIntBSTTree theBalancedBinarySearchTree = new MyIntBSTTree();
		theBalancedBinarySearchTree = theBalancedBinarySearchTree.buildBalancedTree(theArrayOfIntegers);
		theBalancedBinarySearchTree.display();
		
		System.out.println("Height: " + theBalancedBinarySearchTree.height());
		
		System.out.println(
			"Number of comparisons to search " + theIntegerToFindInTheTree + ": " +
			theBalancedBinarySearchTree.comparisons(theIntegerToFindInTheTree)
		);
		
		System.out.println("\n-----------------------------------\n");
		
		System.out.println("Build a worst tree by calling buildWorstTree(arr):");
		MyIntBSTTree theWorstBinarySearchTree = new MyIntBSTTree();
		theWorstBinarySearchTree = theWorstBinarySearchTree.buildWorstTree(theArrayOfIntegers);
		theWorstBinarySearchTree.display();
		
		System.out.println("Height: " + theWorstBinarySearchTree.height());
		System.out.println(
			"Number of comparisons to search " + theIntegerToFindInTheTree + ": " +
			theWorstBinarySearchTree.comparisons(theIntegerToFindInTheTree)
		);
		
		System.out.println("====================\n\n");

	}

}
