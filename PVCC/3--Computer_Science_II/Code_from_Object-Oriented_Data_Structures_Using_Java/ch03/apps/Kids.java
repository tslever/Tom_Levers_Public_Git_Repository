package ch03.apps;
public class Kids
{
  private static int countKids(int girlCount, int boyCount)
  {
    int totalKids;
    totalKids = girlCount + boyCount;
    return(totalKids);
  }

  public static void main(String[] args) 
  { 
    int numGirls;
    int numBoys;
    int numChildren;
    
    numGirls = 12;
    numBoys = 13;
    numChildren = countKids(numGirls, numBoys);
    
    System.out.println("Number of children is " + numChildren);
  } 
}


