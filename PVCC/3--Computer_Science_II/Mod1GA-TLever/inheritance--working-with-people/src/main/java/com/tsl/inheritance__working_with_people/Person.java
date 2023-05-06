package com.tsl.inheritance__working_with_people;


/**
 * Person represents the structure for a person.
 * @author Tom Lever
 * @version 1.0
 * @since 05/18/21
 *
 */
class Person {

	/**
	 * name is an attribute of a person.
	 */
	protected String name;
	
	
	/**
	 * Person() is a zero-argument constructor for Person that initializes a person's name to "".
	 */
	public Person() {
		this.name = "";
	}
	
	
	/**
	 * Person(String name) is a one-argument constructor for Person that initializes a person's name to name.
	 * @param name
	 */
	public Person(String name) {
		this.name = name;
	}
	
	
	/**
	 * getName provides the name of this person.
	 * @return
	 */
	public String getName() {
		return this.name;
	}
	
	
	/**
	 * setName sets the name of this person to name.
	 * @param name
	 */
	public void setName(String name) {
		this.name = name;
	}
	
	
	/**
	 * toString provides information for this person.
	 */
	@Override
	public String toString() {
		return "Name: " + this.name;
	}
	
}
