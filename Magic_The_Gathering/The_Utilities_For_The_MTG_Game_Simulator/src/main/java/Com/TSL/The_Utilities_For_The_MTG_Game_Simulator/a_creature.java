package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;


public class a_creature extends a_permanent {
	
	private boolean Indicator_Of_Whether_This_Creature_Is_Blocked = false;
	private boolean Indicator_Of_Whether_This_Creature_Is_Blocking = false;
	
	public a_creature(String The_Name, boolean The_Tapped_Status) {
		super(The_Name, The_Tapped_Status);
	}
	
	public boolean provides_its_indicator_of_whether_this_creature_is_blocked() {
		return this.Indicator_Of_Whether_This_Creature_Is_Blocked;
	}	
	
	public boolean provides_its_indicator_of_whether_this_creature_is_blocking() {
		return this.Indicator_Of_Whether_This_Creature_Is_Blocking;
	}
}