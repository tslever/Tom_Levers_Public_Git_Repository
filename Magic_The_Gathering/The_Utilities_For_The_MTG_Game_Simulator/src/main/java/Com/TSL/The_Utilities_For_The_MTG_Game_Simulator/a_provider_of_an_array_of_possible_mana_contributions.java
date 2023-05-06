package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;


import java.util.HashMap;


public class a_provider_of_an_array_of_possible_mana_contributions {

	private HashMap<String, a_mana_contribution[]> Dictionary_Of_Permanent_Names_And_Possible_Mana_Contributions = new HashMap<String, a_mana_contribution[]>();
		
	public a_provider_of_an_array_of_possible_mana_contributions() {
		this.Dictionary_Of_Permanent_Names_And_Possible_Mana_Contributions.put("Swamp", new a_mana_contribution[] { new a_mana_contribution(1, 0, 0, 0, 0, 0) });
		this.Dictionary_Of_Permanent_Names_And_Possible_Mana_Contributions.put("Island", new a_mana_contribution[] { new a_mana_contribution(0, 1, 0, 0, 0, 0) });
		this.Dictionary_Of_Permanent_Names_And_Possible_Mana_Contributions.put("Forest", new a_mana_contribution[] { new a_mana_contribution(0, 0, 0, 1, 0, 0) });
		this.Dictionary_Of_Permanent_Names_And_Possible_Mana_Contributions.put("Mountain", new a_mana_contribution[] { new a_mana_contribution(0, 0, 0, 0, 1, 0) });
		this.Dictionary_Of_Permanent_Names_And_Possible_Mana_Contributions.put("Plains", new a_mana_contribution[] { new a_mana_contribution(0, 0, 0, 0, 0, 1) });
	}
	
	public a_mana_contribution[] provides_the_array_of_possible_mana_contributions_corresponding_to(String The_Permanent_Name) {
		return this.Dictionary_Of_Permanent_Names_And_Possible_Mana_Contributions.get(The_Permanent_Name);
	}
	
}