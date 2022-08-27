package ch08.apps;
import ch08.maps.*;

public class MapExample 
{
  public static void main(String[] args)
  {
    boolean result;
    MapInterface<Character, String> example;
    example = new ArrayListMap<Character, String>();
    System.out.println("Expect 'true':\t" + example.isEmpty());
    System.out.println("Expect '0':\t" + example.size()); 
     
    System.out.println("Expect 'null':\t" + example.put('C', "cat"));
    example.put('D', "dog");   example.put('P', "pig");    
    example.put('A', "ant");   example.put('F', "fox");    
    System.out.println("Expect 'false':\t" + example.isEmpty());   
    System.out.println("Expect '5:\t" + example.size());  
    System.out.println("Expect 'true':\t" + example.contains('D'));
    System.out.println("Expect 'false':\t" + example.contains('E'));
    System.out.println("Expect 'dog':\t" + example.get('D'));
    System.out.println("Expect 'null':\t" + example.get('E'));

    System.out.println("Expect 'cat':\t" + example.put('C',"cow"));
    System.out.println("Expect 'cow':\t" + example.get('C'));
    System.out.print("Expect 5 animals: ");
    for (MapEntry<Character,String> m: example)
      System.out.print(m.getValue() + "\t");

    System.out.println("\nExpect 'pig':\t" + example.put('P', null));
    System.out.println("Expect 'dog':\t" + example.remove('D'));
    System.out.print("Expect 3 animals plus a 'null': ");
    for (MapEntry<Character,String> m: example)
      System.out.print(m.getValue() + "\t");
  }
}