//----------------------------------------------------------------------
// CustomerGenerator.java     by Dale/Joyce/Weems              Chapter 4
//
// Generates a sequence of random Customer objects based on the 
// constructor arguments for min and max inter-arrival and service times. 
// Assumes a flat distribution of both inter-arrival and service times.
// Assumes time starts at 0.
//----------------------------------------------------------------------

package support;

import java.util.Random;

public class CustomerGenerator
{
  protected int minIAT;   // minimum inter-arrival time
  protected int maxIAT;   // maximum inter-arrival time
  protected int minST;    // minimum service time
  protected int maxST;    // maximum service time
  
  protected int currTime = 0;   // current time
  
  Random rand = new Random();   // to generate random numbers 

  public CustomerGenerator (int minIAT, int maxIAT, int minST, int maxST)
  // Preconditions: all arguments >= 0
  //                minIAT <= maxIAT
  //                minST  <= maxST
  {
    this.minIAT = minIAT;    this.maxIAT = maxIAT;
    this.minST  = minST;     this.maxST  = maxST;
  }

  public void reset()
  {
    currTime = 0;
  }

  public Customer nextCustomer()
  // Creates and returns the next random customer.
  {
    int IAT;  // next inter-arrival time
    int ST;   // next service time
    
    IAT = minIAT + rand.nextInt(maxIAT - minIAT + 1);
    ST  = minST  + rand.nextInt(maxST - minST + 1);
    
    currTime = currTime + IAT;  // updates current time to the arrival
                                // time of next customer
                                
    Customer next = new Customer(currTime, ST);
    return next;
  }
}
 