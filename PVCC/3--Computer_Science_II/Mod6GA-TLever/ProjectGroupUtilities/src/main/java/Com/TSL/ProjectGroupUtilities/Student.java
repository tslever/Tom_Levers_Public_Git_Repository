package Com.TSL.ProjectGroupUtilities;


/**
 *	Program: 	Student.java
 * 	Scope:		Defines a class Student, representing a student inrolled in CSC 202.
 *
 */

public class Student
{
	//instance variables
	private String studentID;
	// this variable represents credits taken so far, and  helps classification of student as freshman, sophomore, and so on.
	private String name;
	private int number;
	private String grade;

	/**
	 * Constructs the student with the specified information.
	 *
	 * @param studentID - the studentID of the student
	 * @param number  - the number of credits student has taken so far
	 * @param name - the name of the student
	 * @param grade the grade received for a specific assignment
	 */
	public Student(String studentID,  String name,int number, String grade)
	{
		this.studentID = studentID;
		this.name = name;
		this.number = number;
		if (grade == null)
			this.grade = "";
		else
			this.grade = grade;
	}

	/**
	 * Constructs the course with the specified information, with no grade
	 * established, because the student has not submitted the assignment.
	 *
	 * @param studentID the studentID of the student
	 * @param number the number of credits the student has taken so far
	 * @param name the name of the course
	 */
	public Student(String studentID,  String name, int number)
	{
		this(studentID, name,  number,"");
	}

	// accessors

	/**
	 * Returns the studentID of the student.
	 *
	 * @return the studentID of the student
	 */
	public String getStudentID()
	{
		return studentID;
	}


	/**
	 * Returns the name of this student.
	 *
	 * @return the name of the student
	 */
	public String getName()
	{
		return name;
	}
	
   
   /**
	 * Returns the number of credits student has taken so far.
	 *
	 * @return the number of credits student has taken so far
	 */
	public int getNumber()
	{
		return number;
	}

	/**
	 * Returns the grade for this student for a specific assignment
	 *
	 * @return the grade for the assignment
	 */
	public String getGrade()
	{
		return grade;
	}


	// mutators

	/**
	 * Sets the grade for this course to the one specified.
	 *
	 * @param grade the new grade for the course
	 */
	public void setGrade(String grade)
	{
		this.grade = grade;
	}

	/**
	 * Returns true if this has received a grade for the assignment so far
	 *
	 * @return true if the student has a grade so far and false otherwise
	 */
	public boolean graded()
	{
		return !grade.equals("");
	}

	/**
	 * Determines if this student is equal to the one specified, based on the
	 * student (studentID and number).
	 *
	 * @return true if this student is equal to the parameter
	 */
	public boolean equals(Object other)
	{
		boolean result = false;
		if (other instanceof Student)
		{
			Student otherStudent = (Student) other;
			if (studentID.equals(otherStudent.getStudentID()) &&
					number == otherStudent.getNumber())
				result = true;
		}
		return result;
	}

	/**
	 * Creates and returns a string representation of this course.
	 *
	 * @return a string representation of the course
	 */
	public String toString()
	{
		String result ="\n" + name + "\tID: "+ studentID + "\nNumber of credits: " + number;
		if (!grade.equals(""))
			result += "\nGrade: " + grade ;
		return result;
	}
}