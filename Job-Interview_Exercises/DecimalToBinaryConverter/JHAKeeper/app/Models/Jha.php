<?php
// The JhaListing model stores records in the jha_listings table.
// The database table corresponding to JhaListing has a primary key column named id.
// The primary key is an incrementing integer value.
// The database table corresponding to JhaListing contains created_at and updated_at columns.
// created_at and updated_at values are automatically set when models are created or updated.
// Eloquent models use the default database connection.

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class Jha extends Model
{
    use HasFactory;

    protected $table = 'jhas';
    //protected $primaryKey = 'id';
    //public $timestamps = true;

    protected $fillable = [
        'activity_name',
        'job_step',
        'hazard',
        'control'
    ];
}
