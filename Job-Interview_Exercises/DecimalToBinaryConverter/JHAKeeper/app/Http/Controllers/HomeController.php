<?php

namespace App\Http\Controllers;

class HomeController extends Controller
{
    /**
     * Create a new controller instance.
     *
     * @return void
     */
    public function __construct()
    {
        $this->middleware('auth');
    }

    /**
     * When client navigates to endpoint '/home',
     * if the message interface corresponding to endpoint '/home'
     * didn't redirect the client to the '/activities' endpoint,
     * HTML corresponding to app.blade.php and home.blade.php
     * would be returned to the client.
     *
     * @return \Illuminate\Contracts\Support\Renderable
     */
    public function index()
    {
        return view('home');
    }
}
