<?php

namespace App\Http\Controllers;

class ActivitiesController extends Controller
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
     * When client navigates to endpoint '/activities',
     * the message interface associated with '/activities' calls index.
     * Laravel gets $activities, a slice of database table jhas,
     * adds HTML returned by Laravel / PHP commands in activities.blade.php
     * to the HTML already in activities.blade.php,
     * and returns the HTML to the client.
     *
     * @return \Illuminate\Contracts\Support\Renderable
     */
    public function index()
    {
        $activities = \Illuminate\Support\Facades\DB::table('jhas')
            ->select('activity_name')
            ->groupBy('activity_name')
            ->get();
        return view('activities', ['activities' => $activities]);
    }
}
