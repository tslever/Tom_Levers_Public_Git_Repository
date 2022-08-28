package ch04.threads;

public class Increase implements Runnable
{
   private Counter c;
   private int amount;
   
   public Increase (Counter c, int amount)
   {
      this.c = c;  this.amount = amount;
   }
   
   public void run()
   {
      for (int i = 1; i <= amount; i++)
         c.increment();
   }
}