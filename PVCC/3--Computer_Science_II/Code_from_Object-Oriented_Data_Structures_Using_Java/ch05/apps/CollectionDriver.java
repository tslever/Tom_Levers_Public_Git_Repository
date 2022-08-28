package ch05.apps;
import ch05.collections.*;

public class CollectionDriver 
{
  public static void main(String[] args)
  {
    boolean result;
    CollectionInterface<String> test;
    test = new ArrayCollection<String>();
    System.out.println("Expect 'true':\t" + test.isEmpty());   
    System.out.println("Expect 'false':\t" + test.isFull());
    result = test.add("alpha");
    System.out.println("Expect 'true':\t" + result);
    System.out.println("Expect 'alpha':\t" + test.get("alpha"));    
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
    System.out.println(test);
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
    System.out.println(test);
  }
}