@extends('layouts.app')

@section('content')
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-md-8">

                <a href="/">Return to Welcome Page</a><br/>
                <a href="/jha_form">Go to Form for Entering JHA's</a><br/><br/>

                <h1>Links to Job Hazard Analyses for Activities</h1><br/>

                <table width="100%">
                    <tr><th>Activities</th></tr>
                    @foreach($activities as $activity)
                        <tr>
                            <td>
                                <a href="/jha/{{$activity->activity_name}}">
                                    {{$activity->activity_name}}
                                </a>
                            </td>
                        </tr>
                    @endforeach
                </table>

            </div>
        </div>
    </div>
@endsection
