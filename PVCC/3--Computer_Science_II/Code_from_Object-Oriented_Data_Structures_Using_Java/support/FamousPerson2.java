//----------------------------------------------------------------------
// FamousPerson2.java         by Dale/Joyce/Weems              Chapter 8
//
// Supports famous people objects having a first name, last name, a year
// of birth and a short bit of information.
//
// Expands the previously defined FamousPerson class with a hashCode
// method.
//----------------------------------------------------------------------

package support;

import java.util.Comparator;

public class FamousPerson2 implements Comparable<FamousPerson2>
{
  protected String firstName, lastName, fact;
  protected int yearOfBirth;    

  public FamousPerson2(String first, String last, int yob, String f)
  {
    firstName = first;  lastName = last; fact = f; yearOfBirth = yob;
  }

  public String getFirstName() {return firstName ;}
  public String getLastName() {return lastName;}
  public String getFact() {return fact;}
  public int getYearOfBirth() {return yearOfBirth;}

  @Override 
  public boolean equals(Object obj)
  // Returns true if 'obj' is a FamousPerson2 with same first and last 
  // names as this FamousPerson2, otherwise returns false.
  {
     if (obj == this)
        return true;
     else 
     if (obj == null || obj.getClass() != this.getClass())
        return false;
     else
     {
        FamousPerson2 fp = (FamousPerson2) obj; 
        return (this.firstName.equals(fp.firstName) &&
                this.lastName.equals(fp.lastName)); 
     }
  }

  @Override
  public int hashCode()
  // Returns a hash code value for this FamousPerson2 object.
  {
    return Math.abs((lastName.hashCode() * 3) 
                     + firstName.hashCode());
  }
  
  public int compareTo(FamousPerson2 other)
  // Compares this FamousPerson2 with 'other' for order. Returns a 
  // negative integer, zero, or a positive integer as this object 
  // is less than, equal to, or greater than 'other'.
  {
    if (!this.lastName.equals(other.lastName))
      return this.lastName.compareTo(other.lastName);
    else
      return this.firstName.compareTo(other.firstName);
  }

  @Override
  public String toString()
  {
    return (firstName + " " + lastName + "(Born " + yearOfBirth +
            "): " + fact);
  }
  
  public static Comparator<FamousPerson2> yearOfBirthComparator()
  {
    return new Comparator<FamousPerson2>()
    {
       public int compare(FamousPerson2 element1, FamousPerson2 element2)
       {
         return (element1.yearOfBirth - element2.yearOfBirth);
       }
    };
  }

}
 