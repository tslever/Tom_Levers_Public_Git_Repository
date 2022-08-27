package ch05.apps;
import ch05.collections.*;

public class CollectionDriver2
{
  public static void main(String[] args)
  {
    boolean result;
    CollectionInterface<String> test;
    test = new SortedArrayCollection<String>(5);
    result = test.add("beta");
    System.out.println(test);
    result = test.add("alpha");
    System.out.println(test);
    result = test.add("gamma");
System.out.println(test);
    result = test.add("delta");
System.out.println(test);
    result = test.add("epsilon");
System.out.println(test);
    System.out.println("Expect 'false':\t" + test.isEmpty());   
    System.out.println("Expect 'true':\t" + test.isFull());
    result = test.add("pi");
System.out.println(test);
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
System.out.println(test);
    System.out.println("Expect 'true':\t" + result);
    System.out.println("Expect 'false':\t" + test.isEmpty());   
    System.out.println("Expect 'false':\t" + test.isFull());
    result = test.contains("delta");
    System.out.println("Expect 'false':\t" + result);
    System.out.println("Expect '5':\t" + test.size());
System.out.println(test);
    result = test.remove("alpha");
System.out.println(test);
    System.out.println("Expect 'true':\t" + result);
    System.out.println("Expect '4':\t" + test.size());
System.out.println(test);
    result = test.remove("pi");
System.out.println(test);
    System.out.println("Expect 'true':\t" + result);
    System.out.println("Expect '3':\t" + test.size());
result = test.add("aa");
result = test.add("aaa");
result = test.add("aaaa");
result = test.add("aaaaa");
result = test.add("aaaa");
result = test.add("z");
result = test.add("zz");
result = test.add("mmmm");
result = test.add("eee");
result = test.add("e");
result = test.add("a");
result = test.add("cat");
result = test.add("dat");
result = test.add("fat");
result = test.add("gat");
result = test.add("fat");
result = test.add("dat");
result = test.add("cat");
result = test.add("hippo");
result = test.add("hippo");
System.out.println(test);

  }
}