package ch04.concurrency;

import ch04.threads.*;

public class Demo02
{
  public static void main(String[] args) throws InterruptedException
  {
    Counter  c = new Counter();
    Runnable r = new Increase(c, 10000);
    Thread   t = new Thread(r);

    t.start();
    
    System.out.println("Expected: 10000");
    System.out.println("Count is: " + c.getCount());
  }
}