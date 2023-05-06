package com.tsl.person_class;


/**
* @author EMILIA BUTU
* version 1.0
* since   2020-07
*
* Student name:  Tom Lever
* Completion date: 05/17/21
*
* PersonDriver.java
*
* This class represents the driver for class Person. The role of it is to test the methods of Person class.
*/

public class PersonDriver {
	
	/**
	 * main represents the entry point of the program.
	 * main defines and instantiates three objects of type Person, displays information for the three objects, and
	 * adds missing information using mutators.
	 * @param args
	 */
	public static void main(String[] args) {

		//*** Task #1: define and instantiate three objects of type Person, using the three constructors
		Person p1=new Person();
		Person p2=new Person("Jane Young", 32421);
		Person p3=new Person("Ella Jones",54231,"IT");

		
		//*** Task #2: display the information of the three objects of type Person
		System.out.println("The initial information of the persons is: ");
		System.out.println(p1);
		System.out.println();
		System.out.println(p2);
		System.out.println();
		System.out.println(p3);
		System.out.println();

		
		//*** Task #3: add the missing information using mutators
		p1.setName("Jimmy Dean");
		p1.setIdNumber(23123);
		p1.setDepartment("Sales");
		p2.setDepartment("Marketing");
		System.out.println("New information about the persons is: ");
		System.out.println(p1);
		System.out.println();
		System.out.println(p2);
		System.out.println();
		System.out.println(p3);
		System.out.println();
		
	}
	
}