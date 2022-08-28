//---------------------------------------------------------------------------
// SyncCounter.java            by Dale/Joyce/Weems                  Chapter 4
//
// Tracks the current value of a counter.
// Provides synchronized access to the increment method.
//---------------------------------------------------------------------------

package ch04.threads;

public class SyncCounter
{
  private int count;
  
  public SyncCounter()
  {
     count = 0;
  }

  public synchronized void increment()
  {
    count++;
  }

  public int getCount(){return count;}
}
