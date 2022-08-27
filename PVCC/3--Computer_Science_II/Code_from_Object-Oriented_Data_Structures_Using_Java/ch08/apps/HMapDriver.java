package ch08.apps;
import ch08.maps.*;

public class HMapDriver 
{
  public static void main(String[] args)
  {
    boolean result;
    HMap<String, String> test;
    test = new HMap<String, String>(10,0.75);

/*
String s = null;
test.put(s,"value");
test.put("s",null);
System.out.println("Expect 'null':\t" + test.get("s"));
System.out.println("Expect 'true':\t" + test.contains("s"));
test = new ArrayListMap<String, String>();
*/



    System.out.println("Expect 'true':\t" + test.isEmpty());   
    System.out.println("Expect '0':\t" + test.size());  

    System.out.println("Expect 'null':\t" + test.put("1", "One"));
    System.out.println("Expect 'false':\t" + test.isEmpty());   
    System.out.println("Expect '1':\t" + test.size());  

    System.out.println("Expect 'One':\t" + test.put("1", "One"));
    System.out.println("Expect 'false':\t" + test.isEmpty());   
    System.out.println("Expect '1':\t" + test.size());  


    test.put("2", "Two");    
    test.put("3", "Three");    
    test.put("4", "Four");    
    test.put("5", "Five");    
    System.out.println("Expect '5':\t" + test.size());  

    System.out.println("Expect 'Three':\t" + test.put("3", "Three XXX"));
    System.out.println("Expect 'Three XXX':\t" + test.put("3", "Three"));
    System.out.println("Expect '5':\t" + test.size());  

    System.out.println("Expect 'true':\t" + test.contains("5"));
    System.out.println("Expect 'false':\t" + test.contains("6"));

    System.out.println(test);
   
    test.put("d","d");
     
    System.out.println(test);
    System.out.println("Expect 'true':\t" + test.contains("d"));
    System.out.println("Expect 'false':\t" + test.contains("e"));



    System.out.println("Expect 'One':\t" + test.get("1"));
    System.out.println("Expect 'One':\t" + test.get("1"));
    System.out.println("Expect 'Two':\t" + test.get("2"));
    System.out.println("Expect 'Three':\t" + test.get("3"));
    System.out.println("Expect 'Four':\t" + test.get("4"));
    System.out.println("Expect 'Five':\t" + test.get("5"));
    System.out.println("Expect 'null':\t" + test.get("6"));

    test.put("e","e");

    System.out.println(test);

    System.out.println("\nThe Map is:\n");
    for (MapEntry<String,String> m: test)
      System.out.println(m + "\n");

System.out.println(1);
    test.put("f","f");
System.out.println(2);
    System.out.println(test);
System.out.println(3);
    System.out.println("\nThe Map is:\n");
    for (MapEntry<String,String> m: test)
      System.out.println(m + "\n");
System.out.println(4);



  }
}