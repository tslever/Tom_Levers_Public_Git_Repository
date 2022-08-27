//----------------------------------------------------------------------
// SerSong.java           by Dale/Joyce/Weems                  Chapter 6
//
// Supports song objects having a name and a duration.
// Implements Serializable.
//----------------------------------------------------------------------

package support;

import java.io.*;
import java.text.*;

public class SerSong implements Serializable
{
  protected String name;
  protected int duration;    // in seconds
  
  DecimalFormat fmt = new DecimalFormat("00");  // to format seconds

  public SerSong(String name, int seconds)
  {
    this.name = name;
    duration = seconds;
  }

  public SerSong(String name, int minutes, int seconds)
  {
    this.name = name;
    duration = (60 * minutes) + seconds;
  }

  public String getName()
  {
    return name;
  }
  
  public int getDuration()
  {
    return duration;
  }

  public String toString()
  {
  
    return (name + " " + (duration / 60) + ":" 
            + fmt.format(duration % 60));
  }
}
 