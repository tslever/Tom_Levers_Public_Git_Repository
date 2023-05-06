package Com.TSL.LineEditorUtilities;


/** 
 * LLTests.java: sample JUnit test cases for LineList class
 * Run this file on the command line with the jar files provided
 *  
 */ 


import java.io.*;
import java.util.Properties;
import static java.lang.System.*;
import static org.junit.Assert.*;
import org.junit.Test;
//import org.junit.*;


public class LLTests{

   LineList line = new LineList();
   
   @Test 
   public void test1(){      
      assertEquals(0,line.words());
   }

   @Test 
   public void test2(){      
      line.addLine("line 1");
      line.addLine("line 2");
      assertEquals(4,line.words());
   }

   @Test 
   public void test3(){
      line.addLine("line 1");
      line.addLine("line 2",1);
      assertEquals("line 2\nline 1", line.toString());
   }

   @Test 
   public void test4(){
      line.addLine("line 1");
      line.addLine("line 2 ");
      line.empty();
      assertEquals(0,line.words());
   }

   @Test 
   public void test5(){
      line.addLine("line 1");
      line.addLine("line 2");
      assertEquals("line 1\nline 2", line.toString());
   }

   @Test 
   public void test6(){
      line.addLine("zero\tone,two;three.four?five!-six@seven:eight");
      line.addLine("nine ten*eleven?  twenty     end   @? ");
      assertEquals(14,line.words());
   }

   @Test 
   public void test7(){
      line.addLine("zero\tone,two;three.four?five!-six@seven eight");
      line.addLine("nine     ten* ? twenty ");
      line.addLine("line 1");
      line.addLine("line 2");
      assertEquals(4,line.lines());
   }

   @Test 
   public void test8(){
      line.addLine("line 1");
      line.addLine("line 2\nline3\nline4");
      assertEquals(2,line.lines());
   }

   @Test 
   public void test9(){
      line.addLine("line1");
      line.addLine("line2\nline3\nline 4");
      assertEquals(5,line.words());
   }

   @Test 
   public void test10(){
      line.addLine("line 1");
      line.addLine("line 2");
      line.addLine("line 3");
      line.addLine("line 4");
      line.delete(2);
      assertEquals(3,line.lines());
   }

   @Test 
   public void test11(){
      line.addLine("line 1");
      line.addLine("line 2");
      line.addLine("line 3");
      line.addLine("line 4");
      line.delete(4);
      String str = "line 1\nline 2\nline 3";
      assertEquals(str,line.toString());
   }

   @Test 
   public void test12(){
      line.addLine("line 1");
      line.addLine("line 2");
      line.addLine("line 3");
      line.addLine("line 4");
      line.addLine("line 5", 2);
      line.delete(2);
      String str = "line 1\nline 2\nline 3\nline 4";
      assertEquals(str,line.toString());
   }

   @Test 
   public void test13(){
      line.addLine("line 1");
      line.addLine("line 2");
      line.addLine("line 3");
      line.addLine("line 4");
      line.delete(3);
      String str = "line 1\nline 2\nline 3";
      assertEquals(3,line.lines());
   }

   @Test 
   public void test14(){
      line.addLine("line 1");
      line.addLine("line 2");
      line.addLine("line 3");
      line.replace("line", "Line");
      String str = "Line 1\nLine 2\nLine 3";
      assertEquals(str,line.toString());
   }

   @Test 
   public void test15(){
      line.addLine("java IS easy to learn");
      line.addLine("java IS open-source");
      line.addLine("java is fast and powerful");
      line.replace("java", "JAVA");
      line.replace("IS", "is");
      String str = "JAVA is easy to learn\nJAVA is open-source\nJAVA is fast and powerful";
      assertEquals(str,line.toString());

      line.empty();
      assertEquals("",line.toString());
   }

   @Test 
   public void test16(){
      line.addLine("java IS easy to learn");
      line.addLine("java IS open-source");
      line.addLine("java IS fast and powerful");
      line.replace("java IS", "Java is");

      String str = "Java is easy to learn\nJava is open-source\nJava is fast and powerful";
      assertEquals(str,line.toString());
   }

   @Test 
   public void test17(){
      PrintStream stdout = System.out;
      ByteArrayOutputStream outContent = new ByteArrayOutputStream();
      System.setOut(new PrintStream(outContent));
      Properties props = getProperties();
      String lineSep = props.getProperty("line.separator");
      String expected = "Line 3 does not exist."+lineSep;
      
      line.addLine("line 1");
      line.addLine("line 2");
      line.line(3);
      System.setOut(stdout);
      String fromPrint=outContent.toString();
      assertEquals(expected , fromPrint);     
   }

   @Test 
   public void test18(){
      PrintStream stdout = System.out;
      ByteArrayOutputStream outContent = new ByteArrayOutputStream();
      System.setOut(new PrintStream(outContent));
      Properties props = getProperties();
      String lineSep = props.getProperty("line.separator");
      String expected = "line 2"+lineSep;
      
      line.addLine("line 1");
      line.addLine("line 2");
      line.addLine("line 3");
      line.line(2);
      System.setOut(stdout);
      String fromPrint=outContent.toString();
      assertEquals(expected , fromPrint);
   }

   @Test 
   public void test19(){    
      PrintStream stdout = System.out;
      ByteArrayOutputStream outContent = new ByteArrayOutputStream();
      System.setOut(new PrintStream(outContent));
      Properties props = getProperties();
      String lineSep = props.getProperty("line.separator");      
      String expected = "1. line 1"+lineSep+"2. line 2"+lineSep+"3. line 3"+lineSep;
      
      line.addLine("line 1");
      line.addLine("line 2");
      line.addLine("line 3");

      line.print();
      System.setOut(stdout);
      String fromPrint=outContent.toString();
      assertEquals(expected , fromPrint); 
   }

   @Test 
   public void test20(){
      PrintStream stdout = System.out;
      ByteArrayOutputStream outContent = new ByteArrayOutputStream();
      System.setOut(new PrintStream(outContent));
      Properties props = getProperties();
      String lineSep = props.getProperty("line.separator");   
      
      String expected = "1. banana"+lineSep+"2. line 1"+lineSep+"3. line 3"+lineSep+"4. red apple"+lineSep+"5. last line"+lineSep;
      line.addLine("line 1");
      line.addLine("line 2");
      line.addLine("line 3");
      line.delete(2);
      line.addLine("apple");  
      line.addLine("banana", 1);  
      line.replace("apple", "red apple");  
      line.addLine("last line"); 
      
      line.print();
      System.setOut(stdout);
      String fromPrint=outContent.toString();
      assertEquals(expected, fromPrint); 

      PrintStream stdout1 = System.out;
      ByteArrayOutputStream outContent1 = new ByteArrayOutputStream();
      System.setOut(new PrintStream(outContent1));      
      line.line(2);
      System.setOut(stdout1);
      String fromPrint1=outContent1.toString();
      assertEquals("line 1"+lineSep, fromPrint1);
      
      assertEquals(5 , line.lines());
      assertEquals(9 , line.words()); 
      assertEquals("banana\nline 1\nline 3\nred apple\nlast line" , line.toString());    
   }

}