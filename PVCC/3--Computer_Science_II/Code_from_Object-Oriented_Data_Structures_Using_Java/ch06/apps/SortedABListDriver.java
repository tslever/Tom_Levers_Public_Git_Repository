package ch06.apps;

import ch06.lists.*;
import java.util.*;
import support.*;

public class SortedABListDriver 
{
  public static void main(String[] args)
  {
    boolean result;
    String hold;
    SortedABList<String> test;
    Iterator<String> iter;
    Comparator<String> comp = new ReverseStringComparator();
    test = new SortedABList<String>(comp);
    
    // Empty list
    System.out.println("Testing Empty List");
    System.out.println("Expect 'true':\t" + test.isEmpty());   
    System.out.println("Expect 'false':\t" + test.isFull());
    System.out.println("Expect '0':\t" + test.size());
    System.out.println("Expect 'false':\t" + test.remove("alpha"));
    System.out.println("Expect 'false':\t" + test.contains("alpha"));
    System.out.println("Expect 'null':\t" + test.get("alpha"));
    System.out.println("\nTrying iterator should see nothing:");
    for (String x: test)
       System.out.println(x);
    System.out.println("Finished trying iterator.");

    System.out.println("\nTrying to generate exceptions:");
    try {test.add(0, "alpha");} catch(UnsupportedOperationException eob){System.out.println(eob.getMessage());}
    try {test.get(1);} catch(IndexOutOfBoundsException eob){System.out.println(eob.getMessage());}
    try {test.remove(1);} catch(IndexOutOfBoundsException eob){System.out.println(eob.getMessage());}

    // One element list
    System.out.println("Testing One Element List");
//    test = new SortedABList<String>(5);
    test = new SortedABList<String>(comp);

    test.add("alpha");
    System.out.println(test);
//    test = new SortedABList<String>(5);
    test = new SortedABList<String>(comp);


//    test.add(0,"alpha");
    test.add("alpha");

    System.out.println(test);
    System.out.println("Expect 'false':\t" + test.isEmpty());   
    System.out.println("Expect 'false':\t" + test.isFull());
    System.out.println("Expect '1':\t" + test.size());
    System.out.println("Expect 'true':\t" + test.contains("alpha"));
    System.out.println("Expect 'false':\t" + test.contains("beta"));
    System.out.println("Expect 'alpha':\t" + test.get("alpha"));
    System.out.println("Expect 'null':\t" + test.get("beta"));
    System.out.println("\nTrying iterator should see alpha:");
    for (String x: test)
       System.out.println(x);
    System.out.println("Finished trying iterator.");
    System.out.println("Expect 'true':\t" + test.remove("alpha"));
    System.out.println("Expect 'true':\t" + test.isEmpty());   
    System.out.println("Expect 'false':\t" + test.isFull());
    System.out.println("Expect '0':\t" + test.size());
    System.out.println("Expect 'false':\t" + test.contains("alpha"));
    System.out.println("Expect 'null':\t" + test.get("alpha"));
    System.out.println("\nTrying iterator should see nothing:");
    for (String x: test)
       System.out.println(x);
    System.out.println("Finished trying iterator.");
//    test = new SortedABList<String>(5);
    test = new SortedABList<String>(comp);

    test.add("alpha");
    System.out.println(test);
    System.out.println("\nTrying iterator should see alpha:");
    iter = test.iterator();
    while (iter.hasNext())
    {
      hold = iter.next();
      System.out.println(hold);
    }
    System.out.println("Finished trying iterator.");
    System.out.println("\nTrying iterator should see alpha:");
    iter = test.iterator();
    while (iter.hasNext())
    {
      hold = iter.next();
      System.out.println(hold);
      iter.remove();
    }
    System.out.println("Finished trying iterator.");
    System.out.println("\nTrying iterator should see nothing:");
    iter = test.iterator();
    while (iter.hasNext())
    {
      hold = iter.next();
      System.out.println(hold);
    }
    System.out.println("Finished trying iterator.");
    
    // Two element list
    System.out.println("\n\nTesting Two Element Lists");
//    test = new SortedABList<String>(5);
    test = new SortedABList<String>(comp);

    test.add("alpha");
    
//    test.add(1, "beta");
test.add("beta");

    System.out.println(test);
//    test = new SortedABList<String>(5);
    test = new SortedABList<String>(comp);

    
//    test.add(0,"alpha");
test.add("alpha");

    test.add("beta");
    System.out.println(test);
    System.out.println("Expect 'false':\t" + test.isEmpty());   
    System.out.println("Expect 'false':\t" + test.isFull());
    System.out.println("Expect '2':\t" + test.size());
    System.out.println("Expect 'true':\t" + test.contains("alpha"));
    System.out.println("Expect 'true':\t" + test.contains("beta"));
    System.out.println("Expect 'false':\t" + test.contains("gamma"));
    System.out.println("Expect 'alpha':\t" + test.get("alpha"));
    System.out.println("Expect 'beta':\t" + test.get("beta"));
    System.out.println("Expect 'null':\t" + test.get("gamma"));

    System.out.println("Expect '0':\t" + test.indexOf("alpha"));
    System.out.println("Expect '1':\t" + test.indexOf("beta"));
    System.out.println("Expect '-1':\t" + test.indexOf("gamma"));

    System.out.println("\nTrying iterator should see alpha beta:");
    for (String x: test)
       System.out.println(x);
    System.out.println("Finished trying iterator.");

    System.out.println("\nTrying to generate exceptions:");
    try {test.add(3, "alpha");} catch(UnsupportedOperationException eob){System.out.println(eob.getMessage());}
    try {test.set(3, "alpha");} catch(UnsupportedOperationException eob){System.out.println(eob.getMessage());}
    try {test.get(2);} catch(IndexOutOfBoundsException eob){System.out.println(eob.getMessage());}
    try {test.remove(3);} catch(IndexOutOfBoundsException eob){System.out.println(eob.getMessage());}

    System.out.println("Expect 'true':\t" + test.remove("beta"));
    System.out.println("Expect 'false':\t" + test.isEmpty());   
    System.out.println("Expect 'false':\t" + test.isFull());
    System.out.println("Expect '1':\t" + test.size());
    System.out.println("Expect 'false':\t" + test.contains("beta"));
    System.out.println("Expect 'null':\t" + test.get("betaa"));
    System.out.println("Expect '-1':\t" + test.indexOf("beta"));
    System.out.println("\nTrying iterator should see alpha:");
    for (String x: test)
       System.out.println(x);
    System.out.println("Finished trying iterator.");

//    test = new SortedABList<String>(5);
    test = new SortedABList<String>(comp);

    test.add("alpha");  test.add("beta");
    System.out.println(test);
    System.out.println("\nTrying iterator should see alpha beta:");
    iter = test.iterator();
    while (iter.hasNext())
    {
      hold = iter.next();
      System.out.println(hold);
    }
    System.out.println("Finished trying iterator.");
    System.out.println("\nTrying iterator should see alpha beta:");
    iter = test.iterator();
    while (iter.hasNext())
    {
      hold = iter.next();
      System.out.println(hold);
      if (hold.equals("beta")) iter.remove();
    }
    System.out.println("Finished trying iterator.");
    System.out.println("\nTrying iterator should see alpha:");
    iter = test.iterator();
    while (iter.hasNext())
    {
      hold = iter.next();
      System.out.println(hold);
    }
    System.out.println("Finished trying iterator.");

//    test = new SortedABList<String>(5);
    test = new SortedABList<String>(comp);

    test.add("alpha");  test.add("beta");
    System.out.println(test);
    System.out.println("\nTrying iterator should see alpha beta:");
    iter = test.iterator();
    while (iter.hasNext())
    {
      hold = iter.next();
      System.out.println(hold);
      if (hold.equals("alpha")) iter.remove();
    }
    System.out.println("Finished trying iterator.");
    System.out.println("\nTrying iterator should see beta:");
    iter = test.iterator();
    while (iter.hasNext())
    {
      hold = iter.next();
      System.out.println(hold);
    }
    System.out.println("Finished trying iterator.");

    // Multi element list
//    test = new SortedABList<String>(2);
    test = new SortedABList<String>(comp);

    System.out.println("\n\nTesting Multi Element Lists");
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
    
    System.out.println("Expect alpha beta gamma epsilon pi");
    System.out.println(test);
    
    System.out.println("\n\nTrying iterator:");
    for (String x: test)
       System.out.println(x);

    System.out.println("\n\nTrying iterator again and remove epsilon:");
    iter = test.iterator();
    while (iter.hasNext())
    {
      hold = iter.next();
      System.out.println(hold);
      if (hold.equals("epsilon")) iter.remove();
    }

    System.out.println("\n\nTrying iterator:");
    for (String x: test)
       System.out.println(x);

    System.out.println("\n\nTesting sort:");
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




/*
    System.out.println("\n\nTesting indexing:");
    test.add(1, "One");
    test.add(5, "Five");
    for (String x: test)
       System.out.println(x);

    System.out.println("\n\nRemove 0");
    test.remove(0);
    for (String x: test)
       System.out.println(x);

    System.out.println("\n\nRemove 4");
    test.remove(4);
    for (String x: test)
       System.out.println(x);

    System.out.println("\n\nSet 3");
    test.set(3,"Three");
    for (String x: test)
       System.out.println(x);

    System.out.println(test.get(3));
    System.out.println(test.indexOf("Three"));
    System.out.println(test.indexOf("nothing"));
*/    
    
    // Iterator remove
//    test = new SortedABList<String>(15);
    test = new SortedABList<String>(comp);

    System.out.println("\n\nTesting Iterator remove each separately");
    result = test.add("1");
    result = test.add("2");
    result = test.add("3");
    result = test.add("4");

    System.out.println("\n\nTrying iterator 1:");
    iter = test.iterator();
    while (iter.hasNext())
    {
      hold = iter.next();
      System.out.println(hold);
      if (hold.equals("1"))
         iter.remove();
    }

/*
    System.out.println("\n\nTrying iterator:");
    for (String x: test)
       System.out.println(x);
       
    test = new SortedABList<String>(15);
    System.out.println("\n\nTesting Iterator remove each separately");
    result = test.add("1");
    result = test.add("2");
    result = test.add("3");
    result = test.add("4");

    System.out.println("\n\nTrying iterator 2:");
    iter = test.iterator();
    while (iter.hasNext())
    {
      hold = iter.next();
      System.out.println(hold);
      if (hold.equals("2"))
         iter.remove();
    }

    System.out.println("\n\nTrying iterator:");
    for (String x: test)
       System.out.println(x);
       
       
    test = new SortedABList<String>(15);
    System.out.println("\n\nTesting Iterator remove each separately");
    result = test.add("1");
    result = test.add("2");
    result = test.add("3");
    result = test.add("4");

    System.out.println("\n\nTrying iterator 3:");
    iter = test.iterator();
    while (iter.hasNext())
    {
      hold = iter.next();
      System.out.println(hold);
      if (hold.equals("3"))
         iter.remove();
    }

    System.out.println("\n\nTrying iterator:");
    for (String x: test)
       System.out.println(x);
       
       
    test = new SortedABList<String>(15);
    System.out.println("\n\nTesting Iterator remove each separately");
    result = test.add("1");
    result = test.add("2");
    result = test.add("3");
    result = test.add("4");

    System.out.println("\n\nTrying iterator 4:");
    iter = test.iterator();
    while (iter.hasNext())
    {
      hold = iter.next();
      System.out.println(hold);
      if (hold.equals("4"))
         iter.remove();
    }

    System.out.println("\n\nTrying iterator:");
    for (String x: test)
       System.out.println(x);
       
*/       
       
       
  }
}