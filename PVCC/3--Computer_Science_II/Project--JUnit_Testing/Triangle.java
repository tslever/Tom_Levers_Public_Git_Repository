// Triangle class 

public class Triangle {
    // Instance variables for 3 triangle sides
    private String side1;
    private String side2;
    private String side3;

    // Constructors
    public Triangle() {
        // a default triangle
        this("0", "0", "0");
    }
    
    public Triangle(String str1, String str2, String str3) {
        this.side1 = str1;
        this.side2 = str2;
        this.side3 = str3;
    }
    
    // Instance methods
    public String triangleType() {
        
        String type = "";
        
        int s1 = 0, s2 = 0, s3 = 0;
        String errMessage = "";
        try {
            s1 = Integer.parseInt(side1);
        } catch (NumberFormatException e) {
           errMessage += "The side 1 is not an integer number.\n\n";
        }
        try {
            s2 = Integer.parseInt(side2);
        } catch (NumberFormatException e) {
           errMessage += "The side 2 is not an integer number.\n\n";
        }
        try {
            s3 = Integer.parseInt(side3);
        } catch (NumberFormatException e) {
           errMessage += "The side 3 is not an integer number.\n\n";
        }
        
        // Check for a negative side
        if (s1 < 0 || s2 < 0 || s3 < 0) {
               errMessage += "At least one side is negative!\n";            
        }
        
        // Check for vaide sides
        if ((s1 + s2 <= s3) || (s1 + s3 <= s2) || (s2 + s3 <= s1)) {
               errMessage += "Not a valid triangle!\n";
        }  
         
        if (s1 + s2 + s3 > 1000) {
            errMessage += "This triangle is too big.\n";
        } 
               
        if (errMessage.length() > 0) {
           type = errMessage ;   
        } 
        else {            
            if ((s1 == s2) && (s2 == s3)) {
               type = "This is an equilateral triangle. ";
           } 
           else if (( s1 == s2) || (s2 == s3) || (s1 == s3)) {
               type = "This is an isosceles triangle. ";
           } 
           else {
               type = "This is a scalene triangle. ";
           }                 
        }
        
        return type;        
    }     
}
