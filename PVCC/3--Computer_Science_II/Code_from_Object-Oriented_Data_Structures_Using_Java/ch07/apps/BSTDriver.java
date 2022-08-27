package ch07.apps;

import ch07.trees.*;
import support.*;
import java.util.*;   // Iterator, Comparator

public class BSTDriver 
{
  public static void main(String[] args)
  {
    boolean result;
    String hold;
    BinarySearchTree<String> test;
    Iterator<String> iter, iter2;
    Comparator<String> comp = new ReverseStringComparator();

    test = new BinarySearchTree<String>();
   
    // Empty tree
    System.out.println("Testing Empty Tree");
    System.out.println("Expect 'true':\t" + test.isEmpty());   
    System.out.println("Expect 'false':\t" + test.isFull());
    System.out.println("Expect '0':\t" + test.size());
    System.out.println("Expect 'false':\t" + test.remove("alpha"));
    System.out.println("Expect 'false':\t" + test.contains("alpha"));
    System.out.println("Expect 'null':\t" + test.get("alpha"));
    System.out.println("Expect 'null':\t" + test.min());
    System.out.println("Expect 'null':\t" + test.max());
/*    System.out.println("\nTrying iterator should see nothing:");
    for (String x: test)
       System.out.println(x);
    System.out.println("Finished trying iterator.");
*/

    // One element tree
    System.out.println("\nTesting One Element Tree");
    test = new BinarySearchTree<String>();
    System.out.println("Expect 'true':\t"  + test.add("alpha"));
    System.out.println("Expect 'false':\t" + test.isEmpty());   
    System.out.println("Expect 'false':\t" + test.isFull());
    System.out.println("Expect '1':\t" + test.size());
    System.out.println("Expect 'true':\t" + test.contains("alpha"));
    System.out.println("Expect 'false':\t" + test.contains("beta"));
    System.out.println("Expect 'alpha':\t" + test.get("alpha"));
    System.out.println("Expect 'null':\t" + test.get("beta"));
    System.out.println("Expect 'alpha':\t" + test.min());
    System.out.println("Expect 'alpha':\t" + test.max());


/*    System.out.println("\nTrying iterator should see alpha:");
    for (String x: test)
       System.out.println(x);
    System.out.println("Finished trying iterator.");
*/
    System.out.println("Expect 'true':\t" + test.remove("alpha"));
    System.out.println("Expect 'true':\t" + test.isEmpty());   
    System.out.println("Expect 'false':\t" + test.isFull());
    System.out.println("Expect '0':\t" + test.size());
    System.out.println("Expect 'false':\t" + test.contains("alpha"));
    System.out.println("Expect 'null':\t" + test.get("alpha"));
    System.out.println("Expect 'null':\t" + test.min());
    System.out.println("Expect 'null':\t" + test.max());

/*    System.out.println("\nTrying iterator should see nothing:");
    for (String x: test)
       System.out.println(x);
    System.out.println("Finished trying iterator.");
*/

    // Two element tree
    System.out.println("\n\nTesting Two Element BSTs");
    test = new BinarySearchTree<String>();
//    test = new BinarySearchTree<String>(comp);

    test.add("beta");
    test.add("alpha");

    System.out.println("Expect 'false':\t" + test.isEmpty());   
    System.out.println("Expect 'false':\t" + test.isFull());
    System.out.println("Expect '2':\t" + test.size());
    System.out.println("Expect 'true':\t" + test.contains("alpha"));
    System.out.println("Expect 'true':\t" + test.contains("beta"));
    System.out.println("Expect 'false':\t" + test.contains("gamma"));
    System.out.println("Expect 'alpha':\t" + test.get("alpha"));
    System.out.println("Expect 'beta':\t" + test.get("beta"));
    System.out.println("Expect 'null':\t" + test.get("gamma"));
    System.out.println("Expect 'alpha':\t" + test.min());
    System.out.println("Expect 'beta':\t" + test.max());

/*
    System.out.println("\nTrying iterator should see alpha beta:");
    for (String x: test)
       System.out.println(x);
    System.out.println("Finished trying iterator.");
*/
    System.out.println("Expect 'true':\t" + test.remove("beta"));
    System.out.println("Expect 'false':\t" + test.isEmpty());   
    System.out.println("Expect 'false':\t" + test.isFull());
    System.out.println("Expect '1':\t" + test.size());
    System.out.println("Expect 'false':\t" + test.contains("beta"));
    System.out.println("Expect 'null':\t" + test.get("beta"));

/*
    System.out.println("\nTrying iterator should see alpha:");
    for (String x: test)
       System.out.println(x);
    System.out.println("Finished trying iterator.");
*/

    test = new BinarySearchTree<String>();
//    test = new BinarySearchTree<String>(comp);

    test.add("beta");
    test.add("alpha");

    System.out.println("Expect 'false':\t" + test.isEmpty());   
    System.out.println("Expect 'false':\t" + test.isFull());
    System.out.println("Expect '2':\t" + test.size());
    System.out.println("Expect 'true':\t" + test.contains("alpha"));
    System.out.println("Expect 'true':\t" + test.contains("beta"));
    System.out.println("Expect 'false':\t" + test.contains("gamma"));
    System.out.println("Expect 'alpha':\t" + test.get("alpha"));
    System.out.println("Expect 'beta':\t" + test.get("beta"));
    System.out.println("Expect 'null':\t" + test.get("gamma"));
    System.out.println("Expect 'alpha':\t" + test.min());
    System.out.println("Expect 'beta':\t" + test.max());

/*
    System.out.println("\nTrying iterator should see alpha beta:");
    for (String x: test)
       System.out.println(x);
    System.out.println("Finished trying iterator.");
*/
    System.out.println("Expect 'true':\t" + test.remove("beta"));
    System.out.println("Expect 'false':\t" + test.isEmpty());   
    System.out.println("Expect 'false':\t" + test.isFull());
    System.out.println("Expect '1':\t" + test.size());
    System.out.println("Expect 'false':\t" + test.contains("beta"));
    System.out.println("Expect 'null':\t" + test.get("beta"));






    // Multi element tree
    test = new BinarySearchTree<String>();

    System.out.println("\n\nTesting Multi Element Trees");
    result = test.add("alpha");
    System.out.println("Expect 'true':\t" + result);
    result = test.contains("alpha");
    System.out.println("Expect 'true':\t" + result);
    result = test.contains("beta");
    System.out.println("Expect 'false':\t" + result);
    result = test.add("beta");
    result = test.add("gamma");
    result = test.add("delta");
    result = test.add("epsilon");
    System.out.println("Expect 'false':\t" + test.isEmpty());   
    System.out.println("Expect 'false':\t" + test.isFull());
    result = test.add("pi");
    System.out.println("Expect 'true':\t" + result);
    result = test.contains("beta");
    System.out.println("Expect 'true':\t" + result);
    result = test.contains("epsilon");
    System.out.println("Expect 'true':\t" + result);
    result = test.contains("pi");
    System.out.println("Expect 'true':\t" + result);
    result = test.remove("magma");
    System.out.println("Expect 'false':\t" + result);
    result = test.remove("delta");
    System.out.println("Expect 'true':\t" + result);
    System.out.println("Expect 'false':\t" + test.isEmpty());   
    System.out.println("Expect 'false':\t" + test.isFull());
    result = test.contains("delta");
    System.out.println("Expect 'false':\t" + result);
    System.out.println("Expect '5':\t" + test.size());
    System.out.println("Expect 'alpha':\t" + test.min());
    System.out.println("Expect 'pi':\t" + test.max());


    test.add("aaa"); test.add("comma"); test.add("ark");
   
    System.out.println("\nExpect \naaa alpha ark beta comma epsilon gamma pi");
    iter = test.getIterator(BSTInterface.Traversal.Inorder);
    while (iter.hasNext())
    {
      hold = iter.next();
      System.out.print(hold + " ");
    }
    System.out.println();

    System.out.println("\nExpect \nalpha aaa beta ark gamma epsilon comma pi");
    iter = test.getIterator(BSTInterface.Traversal.Preorder);
    while (iter.hasNext())
    {
      hold = iter.next();
      System.out.print(hold + " ");
    }
    System.out.println();

    System.out.println("\nExpect \naaa ark comma epsilon pi gamma beta alpha");
    iter = test.getIterator(BSTInterface.Traversal.Postorder);
    while (iter.hasNext())
    {
      hold = iter.next();
      System.out.print(hold + " ");
    }
    System.out.println();

    System.out.println("\nExpect previous two intertwined");
    iter = test.getIterator(BSTInterface.Traversal.Preorder);
    iter2 = test.getIterator(BSTInterface.Traversal.Postorder);
    while (iter.hasNext())
    {
      hold = iter.next();
      System.out.print(hold + " ");
      hold = iter2.next();
      System.out.print(hold + " ");
    }
    System.out.println();

    System.out.println("\n\nTesting sort and for each");
    test.add("a");
    test.add("bb");
    test.add("zzz");
    test.add("aaa");
    test.add("ggggg");
    test.add("eeee");
    test.add("bat");
    test.add("eat");
    test.add("eat");
    test.add("ggggg");
    test.add("eeee");
    test.add("bat");
    test.add("eat");
    test.add("eat");
    test.add("ggggg");
    test.add("eeee");
    test.add("bat");
    test.add("eat");
    test.add("eat");
    test.add("rrrrrr");
    test.add("sss");
    test.add("7");
    test.add("1");
    test.add("6");
    test.add("3");
    test.add("9");
    test.add("5");
    test.add("2");
    test.add("8");
    test.add("4");

    for (String x: test)
       System.out.println(x);

    // Multi element tree
//    test = new BinarySearchTree<String>(new ReverseStringComparator());
//    System.out.println("\n\nTesting Multi Element Trees Reversed");

    test = new BinarySearchTree<String>();
    System.out.println("\n\nTesting Multi Element Trees");
    

    result = test.add("alpha");
    System.out.println("Expect 'true':\t" + result);
    result = test.contains("alpha");
    System.out.println("Expect 'true':\t" + result);
    result = test.contains("beta");
    System.out.println("Expect 'false':\t" + result);
    result = test.add("beta");
    result = test.add("gamma");
    result = test.add("delta");
    result = test.add("epsilon");
    System.out.println("Expect 'false':\t" + test.isEmpty());   
    System.out.println("Expect 'false':\t" + test.isFull());
    result = test.add("pi");
    System.out.println("Expect 'true':\t" + result);
    result = test.contains("beta");
    System.out.println("Expect 'true':\t" + result);
    result = test.contains("epsilon");
    System.out.println("Expect 'true':\t" + result);
    result = test.contains("pi");
    System.out.println("Expect 'true':\t" + result);
    result = test.remove("magma");
    System.out.println("Expect 'false':\t" + result);
    result = test.remove("delta");
    System.out.println("Expect 'true':\t" + result);
    System.out.println("Expect 'false':\t" + test.isEmpty());   
    System.out.println("Expect 'false':\t" + test.isFull());
    result = test.contains("delta");
    System.out.println("Expect 'false':\t" + result);
    System.out.println("Expect '5':\t" + test.size());
    System.out.println("Expect 'alpha':\t" + test.min());
    System.out.println("Expect 'pi':\t" + test.max());


    test.add("aaa"); test.add("comma"); test.add("ark");
   
    System.out.println("\nExpect \naaa alpha ark beta comma epsilon gamma pi");
    iter = test.getIterator(BSTInterface.Traversal.Inorder);
    while (iter.hasNext())
    {
      hold = iter.next();
      System.out.print(hold + " ");
    }
    System.out.println();

    System.out.println("\nExpect \nalpha aaa beta ark gamma epsilon comma pi");
    iter = test.getIterator(BSTInterface.Traversal.Preorder);
    while (iter.hasNext())
    {
      hold = iter.next();
      System.out.print(hold + " ");
    }
    System.out.println();

    System.out.println("\nExpect \naaa ark comma epsilon pi gamma beta alpha");
    iter = test.getIterator(BSTInterface.Traversal.Postorder);
    while (iter.hasNext())
    {
      hold = iter.next();
      System.out.print(hold + " ");
    }
    System.out.println();

    System.out.println("\nExpect previous two intertwined");
    iter = test.getIterator(BSTInterface.Traversal.Preorder);
    iter2 = test.getIterator(BSTInterface.Traversal.Postorder);
    while (iter.hasNext())
    {
      hold = iter.next();
      System.out.print(hold + " ");
      hold = iter2.next();
      System.out.print(hold + " ");
    }
    System.out.println();

    System.out.println("\n\nTesting sort and for each");
    test.add("a");
    test.add("bb");
    test.add("zzz");
    test.add("aaa");
    test.add("ggggg");
    test.add("eeee");
    test.add("bat");
    test.add("eat");
    test.add("eat");
    test.add("ggggg");
    test.add("eeee");
    test.add("bat");
    test.add("eat");
    test.add("eat");
    test.add("ggggg");
    test.add("eeee");
    test.add("bat");
    test.add("eat");
    test.add("eat");
    test.add("rrrrrr");
    test.add("sss");
    test.add("7");
    test.add("1");
    test.add("6");
    test.add("3");
    test.add("9");
    test.add("5");
    test.add("2");
    test.add("8");
    test.add("4");

    for (String x: test)
       System.out.println(x);
  }
}