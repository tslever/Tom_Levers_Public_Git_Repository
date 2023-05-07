package Com.TSL.UtilitiesForUsingTheMapAdt;


/**
* @author EMILIA BUTU
* version 1.0
* since   2020-06
*
* Student name:  Tom Lever
* Completion date: 07/10/21
*
* Demonstrates the use of methods offered by MapInterface, with an implementation of the map using
* array lists.
*
* The file contains basic operations related to Map ADT.
*/

public class MapExample
{
	
	/**
	 * main is the entry point of this program, which demonstrates the use of methods offered by MapInterface.
	 * @param args
	 */
	
	public static void main(String[] args)
	{
		//*** Task #1: declare a variable of MapInterface type, with type of elements of your choice
		MapInterface<Character, String> the_map_of_letters_and_animals;
		
		//*** Task #2: instantiate the variable declared above using the ArrayListMap constructor
		the_map_of_letters_and_animals = new ArrayListMap<Character, String>();

		//*** Task #3: check if the map is  empty, and print out the answer you get 
		System.out.println("Is the map empty (expect \"true\")? " + the_map_of_letters_and_animals.isEmpty());
		
		//*** Task #4: use the appropriate method, and display the size of the map 
		System.out.println("What is the number of entries in the map (expect '0')? " + the_map_of_letters_and_animals.size());
		
		//*** Task #5: use the appropriate method to populate the map with 4-5 entries
		System.out.println("Putting dog in map (expect \"null\"). " + the_map_of_letters_and_animals.put('C', "cat"));
		the_map_of_letters_and_animals.put('D', "dog");
		the_map_of_letters_and_animals.put('P', "pig");
		the_map_of_letters_and_animals.put('A', "ant");
		the_map_of_letters_and_animals.put('F', "fox");

		//*** Task #6: check if the map is  empty, and print out the answer you get
		System.out.println("Is the map empty (expect \"false\")? " + the_map_of_letters_and_animals.isEmpty());
		
		//*** Task #7: use the appropriate method, and display the size of the map
		System.out.println("What is the number of entries in the map (expect '5')? " + the_map_of_letters_and_animals.size());
		
		//*** Task #8: check if certain values belong to the map, and replace some of them
		System.out.println("Checking to see whether an entry corresponding to letter 'D' is in the map (expect \"true\"). " + the_map_of_letters_and_animals.contains('D'));
		System.out.println("Checking to see whether an entry corresponding to letter 'E' is in the map (expect \"false\"). " + the_map_of_letters_and_animals.contains('E'));
		System.out.println("Getting the animal corresponding to letter 'D' (expect \"dog\"). " + the_map_of_letters_and_animals.get('D'));
		System.out.println("Getting the animal corresponding to letter 'E' (expect \"null\"). " + the_map_of_letters_and_animals.get('E'));
		System.out.println("Replacing \"cat\" with \"cow\" (expect \"cat\"). " + the_map_of_letters_and_animals.put('C', "cow"));
		System.out.println("Getting the animal corresponding to letter 'C' (expect \"cow\"). " + the_map_of_letters_and_animals.get('C'));

		//*** Task #9: display the content of the map
		System.out.print("Displaying the animals in the map (expect \"dog pig ant fox cow\").");
		for (MapEntry<Character, String> the_map_entry : the_map_of_letters_and_animals) {
			System.out.print(" " + the_map_entry.getValue());
		}
		System.out.println();
		
		//*** Task #10: remove a number of elements from the map
		System.out.println("Replacing \"pig\" with null (expect \"pig\"). " + the_map_of_letters_and_animals.put('P', null));
		System.out.println("Removing \"dog\" (expect \"dog\"). " + the_map_of_letters_and_animals.remove('D'));
		
		System.out.print("Displaying the animals in the map (expect \"ant fox cow null\".");
		for (MapEntry<Character, String> the_map_entry : the_map_of_letters_and_animals) {
			System.out.print(" " + the_map_entry.getValue());
		}
		System.out.println();
		
	}
}