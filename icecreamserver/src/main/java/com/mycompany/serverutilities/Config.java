/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.serverutilities;

/**
 *
 * @author thoma
 */
public class Config {
    
    public Config() {
        
    }
    
    public MessageInterface[] getMessageInterfaces() {
        
        // A message handler is a bridge to go between message interface and
        // the controller. The message interface needs HTTP-message endpoint
        // and an object that processes the message.
        MessageInterface one = new MessageInterface(
            "/one", new MessageHandler( new Controller() ));
        
        MessageInterface two = new MessageInterface(
            "/two", new MessageHandler( new Controller() ));
        
        return new MessageInterface[]{one, two};
        
    }
    
}
