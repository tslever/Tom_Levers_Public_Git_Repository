package Com.TSL.BinarySearchUtilities;

/**
 * Hello world!
 *
 */
public class App 
{
	
	static int[] theArrayToSearch;
	
	
    public static void main( String[] args )
    {
    	
    	int theIntegerForWhichToSearch = 20;
    	
    	theArrayToSearch = new int[] {4, 6, 7, 15, 20, 22, 25, 27};
        
    	System.out.println(binarySearchRecursivelyFor(theIntegerForWhichToSearch, 0, theArrayToSearch.length));
    	
    	System.out.println(binarySearchIterativelyFor(theIntegerForWhichToSearch, 0, theArrayToSearch.length));
        
    }
    
    
    static boolean binarySearchRecursivelyFor(int target, int first, int last) {
    	
    	if (first > last) {
    		return false;
    	}
    	
    	int midpoint = first + (last - first) / 2; // = (first + last) / 2;
    	
    	if (target == theArrayToSearch[midpoint]) {
    		return true;
    	}
    	
    	if (target > theArrayToSearch[midpoint]) {
    		return binarySearchRecursivelyFor(target, midpoint + 1, last);
    	}
    	else {
    		return binarySearchRecursivelyFor(target, first, midpoint - 1);
    	}
    	
    }
    
    
    static boolean binarySearchIterativelyFor(int target, int first, int last) {
    	
    	int midpoint;
    	
    	while (first <= last) {
    		
    		midpoint = first + (last - first) / 2; // = (first + last) / 2;
    		
    		if (target == theArrayToSearch[midpoint]) {
    			return true;    			
    		}
    		
    		if (target > theArrayToSearch[midpoint]) {
    			first = midpoint + 1;
    		}
    		else {
    			last = midpoint - 1;
    		}
    		
    	}
    	
    	return false;
    	
    }
    
}
