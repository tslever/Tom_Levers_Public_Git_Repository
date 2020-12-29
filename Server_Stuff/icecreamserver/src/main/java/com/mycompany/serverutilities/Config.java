
// Allows executable to find class Config.
package com.mycompany.serverutilities;

/**
 * Defines class Config that encapsulates functionality to get an array of
 * message interfaces.
 * @version 0.0
 * @author Tom Lever
 */
public class Config {
    
    public Config() {
        
    }
    
    /**
     * Defines method getMessageInterfaces, which allows a calling method to
     * get an array of message interfaces.
     * @return new MessageInterface[]{one, two};
     */
    public MessageInterface[] getMessageInterfaces() {
        System.out.println("Config.getMessageInterfaces: Started.");
        
        MessageInterface one = new MessageInterface(
            "/one", new MessageHandler( new Controller() ));
        MessageInterface two = new MessageInterface(
            "/two", new MessageHandler( new Controller() ));
        System.out.println(
            "Config.getMessageInterfaces: Defined message interfaces.");
        
        System.out.println(
            "Config.getMessageInterfaces: Returning array of defined message " +
            "interfaces.");
        return new MessageInterface[]{one, two};
    }
}
