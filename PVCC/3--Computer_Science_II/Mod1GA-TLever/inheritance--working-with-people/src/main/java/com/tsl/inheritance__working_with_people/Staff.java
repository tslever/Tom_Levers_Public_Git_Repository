package com.tsl.inheritance__working_with_people;


/**
* @author EMILIA BUTU
* version 1.0
* since   2020-07
*
* Student name:  Tom Lever
* Completion date: 05/17/21
*
* Staff.java
*
* Driver for the inheritance application testing the inheritance, and
* showing how polymorphism and dynamic binding works.
*/

public class Staff {
	
	/**
	 * main represents the entry point of the program.
	 * main defines and instantiates a group of people and outputs information for each person in the group.
	 * @param args
	 */
	public static void main(String[] args) {
		
		Person[] group = new Person[5];
		group[0] = new Student("Mary Jane", 234);
		group[1] = new Person("Joe Smith");
		group[2] = new Employee("Anna Smiley", 23234);
		group[3] = new Faculty("Jane Dane", 2343, "Lecturer");
		group[4] = new Undergraduate("Edward Stone", 121, "Business");

		for(Person p: group)
		{
			System.out.println(p);
			System.out.println();
		}

	}

}