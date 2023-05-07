package com.tsl.inheritance__more_people;


import org.junit.jupiter.api.Test;


/**
 * PersonTest encapsulates JUnit tests of core functionality of the methods
 * calculateMonthlyGrossPay(), calculateMonthlyGrossPay(int weeklyHoursPlaceholderOrCreditsPerSemester), and
 * displayPayInfo() of class Person.
 * @author Tom Lever
 * @version 1.0
 * @since 05/21/21
 *
 */
public class PersonTest {
	
	
	/**
	 * testCalculateMonthlyGrossPaySufficientlyImplementedWithNoArgument tests calculateMonthlyGrossPay() by displaying
	 * a valid monthly gross pay when Person's calculateMonthlyGrossPay() method is sufficiently implemented.
	 */
	@Test
	public void testCalculateMonthlyGrossPaySufficientlyImplementedWithNoArgument() {
		
		System.out.println("Running testCalculateMonthlyGrossPaySufficientlyImplementedWithNoArgument.");
		
		Person person = new Person() {
			
			private double hourlySalary = 20.00;
			
			@Override
			public double calculateMonthlyGrossPay() {
				return
					this.hourlySalary /*dollars per hour*/ *
					40.0 /*hours per week*/ *
					4.0 /*weeks per month*/
				;
			}
			
		};
		
		try {
			System.out.println("Monthly gross pay: " + person.calculateMonthlyGrossPay());
		}
		catch (ANotSufficientlyImplementedException theNotSufficientlyImplementedException) {
			System.out.println(theNotSufficientlyImplementedException.getMessage());
		}
		
		System.out.println();
			
	}
	
	
	/**
	 * testCalculateMonthlyGrossPaySufficientlyImplementedWithOneArgument tests
	 * calculateMonthlyGrossPay(int weeklyHoursPlaceholderOrCreditsPerSemester) by displaying a valid monthly gross pay
	 * when Person's calculateMonthlyGrossPay(int weeklyHoursPlaceholderOrCreditsPerSemester) method is sufficiently
	 * implemented.
	 */
	@Test
	public void testCalculateMonthlyGrossPaySufficientlyImplementedWithOneArgument() {
		
		System.out.println("Running testCalculateMonthlyGrossPaySufficientlyImplementedWithOneArgument.");
		
		Person person = new Person() {
			
			private double hourlySalary = 20.00;
			
			@Override
			public double calculateMonthlyGrossPay(int weeklyHours) {
				return
					this.hourlySalary /*dollars per hour*/ *
					(double)weeklyHours /*hours per week*/ *
					4.0 /*weeks per month*/
				;
			}
			
		};
		
		try {
			System.out.println("Monthly gross pay: " + person.calculateMonthlyGrossPay(40));
		}
		catch (ANotSufficientlyImplementedException theNotSufficientlyImplementedException) {
			System.out.println(theNotSufficientlyImplementedException.getMessage());
		}
		
		System.out.println();
			
	}
	
	
	/**
	 * testCalculateMonthlyGrossPaySufficientlyImplementedWithNoArgument tests testDisplayPayInfoSufficientlyImplemented
	 * by displaying valid pay info when Person's displayPayInfo method is sufficiently implemented.
	 */
	@Test
	public void testDisplayPayInfoSufficientlyImplemented() {
		
		System.out.println("Running testDisplayInfoSufficientlyImplemented.");
		
		Person person = new Person() {
			
			private double hourlySalary = 20.00;
			
			@Override
			public double calculateMonthlyGrossPay(int weeklyHours) {
				return
					this.hourlySalary /*dollars per hour*/ *
					(double)weeklyHours /*hours per week*/ *
					4.0 /*weeks per month*/
				;
			}
			
			@Override
			public void displayPayInfo() {
				
				int weeklyHours = 40;
				
				System.out.println(
					"Pay is done on an hourly basis.\n" +
					"Number of hours per week: " + weeklyHours + "\n" +
					"Pay per hour: " + this.hourlySalary + "\n" +
					"Monthly salary: " + calculateMonthlyGrossPay(weeklyHours) + "\n"
				);
				
			}
			
		};
		
		try {
			person.displayPayInfo();
		}
		catch (ANotSufficientlyImplementedException theNotSufficientlyImplementedException) {
			System.out.println(theNotSufficientlyImplementedException.getMessage());
		}
		
		System.out.println();
			
	}
	
	
	/**
	 * testCalculateMonthlyGrossPayNotSufficientlyImplementedWithNoArgument tests calculateMonthlyGrossPay() by
	 * displaying the message of ANotSufficientlyImplementedException thrown when Person's calculateMonthlyGrossPay()
	 * method is called before it is sufficiently implemented.
	 */
	@Test
	public void testCalculateMonthlyGrossPayNotSufficientlyImplementedWithNoArgument() {
		
		System.out.println("Running testCalculateMonthlyGrossPayNotSufficientlyImplementedWithNoArgument.");
		
		Person person = new Person();
		
		try {
			System.out.println("Monthly gross pay: " + person.calculateMonthlyGrossPay());
		}
		catch (ANotSufficientlyImplementedException theNotSufficientlyImplementedException) {
			System.out.println(theNotSufficientlyImplementedException.getMessage());
		}
		
		System.out.println();
			
	}
	
	
	/**
	 * testCalculateMonthlyGrossPayNotSufficientlyImplementedWithOneArgument tests
	 * calculateMonthlyGrossPay(int weeklyHoursPlaceholderOrCreditsPerSemester) by displaying the message of
	 * ANotSufficientlyImplementedException thrown when Person's
	 * calculateMonthlyGrossPay(int weeklyHoursPlaceholderOrCreditsPerSemester) method is called before it is
	 * sufficiently implemented.
	 */
	@Test
	public void testCalculateMonthlyGrossPayNotSufficientlyImplementedWithOneArgument() {
		
		System.out.println("Running testCalculateMonthlyGrossPayNotSufficientlyImplementedWithOneArgument.");
		
		Person person = new Person();
		
		try {
			System.out.println("Monthly gross pay: " + person.calculateMonthlyGrossPay());
		}
		catch (ANotSufficientlyImplementedException theNotSufficientlyImplementedException) {
			System.out.println(theNotSufficientlyImplementedException.getMessage());
		}
		
		System.out.println();
			
	}
	
	
	/**
	 * testDisplayPayInfoNotSufficientlyImplemented tests displayPayInfo by displaying the message of
	 * ANotSufficientlyImplementedException thrown when Person's displayPayInfo method is called before it is
	 * sufficiently implemented.
	 */
	@Test
	public void testDisplayPayInfoNotSufficientlyImplemented() {
		
		System.out.println("Running testDisplayPayInfoNotSufficientlyImplemented.");
		
		Person person = new Person();
		
		try {
			person.displayPayInfo();
		}
		catch (ANotSufficientlyImplementedException theNotSufficientlyImplementedException) {
			System.out.println(theNotSufficientlyImplementedException.getMessage());
		}
		
		System.out.println();
			
	}
	
	
}
