package ch04.threads;

public class IncreaseSync implements Runnable
{
   private SyncCounter c;
   private int amount;
   
   public IncreaseSync (SyncCounter c, int amount)
   {
      this.c = c;  this.amount = amount;
   }
   
   public void run()
   {
      for (int i = 1; i <= amount; i++)
         c.increment();
   }
}