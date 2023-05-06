<?php

namespace App\Http\Controllers;

class JhasForActivityController extends Controller
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
     * When client navigates to endpoint '/jha/{activity_name}',
     * the message interface associated with '/jha/{activity_name}' calls index.
     * Laravel gets $jhas, a slice of database table jhas,
     * adds HTML returned by Laravel / PHP commands in jhasforactivity.blade.php
     * to the HTML already in jhasforactivity.blade.php,
     * and returns the HTML to the client.
     *
     * @return \Illuminate\Contracts\Support\Renderable
     */
    public function index($activity_name)
    {
        $jhas = \Illuminate\Support\Facades\DB::table('jhas')
            ->where('activity_name', '=', $activity_name)
            ->orderBy('activity_name')
            ->orderBy('job_step')
            ->orderBy('hazard')
            ->orderBy('control')
            ->get();
        return view('jhasforactivity', ['jhas' => $jhas, 'activity_name' => $activity_name]);
    }
}
