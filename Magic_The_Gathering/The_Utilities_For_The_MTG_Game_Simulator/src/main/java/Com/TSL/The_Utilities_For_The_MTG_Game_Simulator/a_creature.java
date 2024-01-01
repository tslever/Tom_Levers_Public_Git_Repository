package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;


public class a_creature extends a_permanent {
	
	private boolean Indicator_Of_Whether_This_Creature_Is_Blocked = false;
	private boolean Indicator_Of_Whether_This_Creature_Is_Blocking = false;
	
	public a_creature(String The_Name, boolean The_Tapped_Status) {
		super(The_Name);
	}
	
	public boolean is_blocked() {
		return this.Indicator_Of_Whether_This_Creature_Is_Blocked;
	}	
	
	public boolean is_blocking() {
		return this.Indicator_Of_Whether_This_Creature_Is_Blocking;
	}
}