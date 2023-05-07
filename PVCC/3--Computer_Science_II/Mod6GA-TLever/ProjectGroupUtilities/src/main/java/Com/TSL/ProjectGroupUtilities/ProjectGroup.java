package Com.TSL.ProjectGroupUtilities;


/**
* @author EMILIA BUTU
* version 1.0
* since   2020-03 
*
* Student name:  Tom Lever
* Completion date: 06/20/21
*
* ProjectGroup.txt: template file of ProjectGroup.java
* Student tasks: complete tasks specified in the file
*/

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;


/**
 * 
 * Represents a list of students for a project group.
 *
 */

public class ProjectGroup implements Iterable<Student>
{
	
	// instance variable
	private List<Student> list;

	
	/**
	 * Constructs an initially empty list representing a group project.
	 */
	
	public ProjectGroup()
	{
		//*** Task #1: add code here for the constructor
		
		this.list = new LinkedList<Student>();
	}

	
	/**
	 * Adds the specified student to the end of the student list.
	 *
	 * @param student as the student to add
	 */
	
	public void addStudent(Student student) {
		//*** Task #2: create the body of this method
	
		this.list.add(student);
		
	}
	
	
	/**
	 * Finds and returns the student matching the specified studentID
    * @param studentID, as the studentID of the target student
	 * @return the student, or null if not found
	 */
	
	public Student find(String studentID) {
		//*** Task #3: fill in the body of method find
		
		for (Student student : this.list) {
			if (student.getStudentID() == studentID) {
				return student;
			}
		}
		
		return null;
		
	}
	

	/**
	 * Adds the specified student after the target student. 
    * Does nothing if either student is null or if the target is not found.
	 *
	 * @param target the student after which the new student will be added
	 * @param newStudent the student to add
	 */
	
	public void addStudentAfter(Student target, Student newStudent) {
		//*** Task #4: ill in the body of method addStudentAfter

		if ((target == null) || (newStudent == null)) {
			return;
		}
		
		int theIndexOfTheTargetStudent = this.list.indexOf(target);
		
		if (theIndexOfTheTargetStudent == -1) {
			return;
		}
		
		this.list.add(theIndexOfTheTargetStudent + 1, newStudent);
		
	}


	/**
	 * Replaces the specified target student with the new student. Does nothing if
	 * either student is null or if the target is not found.
	 *
	 * @param target the student to be replaced
	 * @param newStudent the new student to add
	 */
	public void replace(Student target, Student newStudent) {
		//*** Task #5: fill in the body of method replace

		if ((target == null) || (newStudent == null)) {
			return;
		}
		
		int theIndexOfTheTargetStudent = this.list.indexOf(target);
		
		if (theIndexOfTheTargetStudent == -1) {
			return;
		}
		
		this.list.set(theIndexOfTheTargetStudent, newStudent);
		
	}
	

	/**
	 * Creates and returns a string representation of this ProjectGroup object.
	 *
	 * @return a string representation of the ProjectGroup object
	 */
	public String toString() {
  		//*** Task #6: fill in the body of the method toString()
	
		String theRepresentationOfThisProjectGroup = "";
		
		for (Student student : this.list) {
			theRepresentationOfThisProjectGroup += student + "\n";
		}
		
		return theRepresentationOfThisProjectGroup;
		
	}
	

	/**
	 * Returns an iterator for this Program of Study.
	 *
	 * @return an iterator for the Program of Study
	 */
	public Iterator<Student> iterator()
	{
		return list.iterator();
	}

}