//----------------------------------------------------------------------
// Customer.java          by Dale/Joyce/Weems                  Chapter 4
//
// Supports customer objects having arrival, service, and finish time
// attributes. Responsible for computing and returning wait time.
//
// Clients should not request wait time unless finish time has been set.
//----------------------------------------------------------------------

package support;

public class Customer
{
  protected int arrivalTime;
  protected int serviceTime;
  protected int finishTime;

  public Customer(int arrivalTime, int serviceTime)
  {
    this.arrivalTime = arrivalTime;
    this.serviceTime = serviceTime;
  }

  public int getArrivalTime(){return arrivalTime;}
  public int getServiceTime(){return serviceTime;}
  public void setFinishTime(int time){finishTime = time;}
  public int getFinishTime(){return finishTime;}

  public int getWaitTime()
  {
    return (finishTime - arrivalTime - serviceTime);
  }

}
 